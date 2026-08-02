import os
import joblib
import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not installed. Skipping deep learning models.")
    DEEP_LEARNING_AVAILABLE = False

# Each real forecast horizon and the ground-truth target column it trains
# against (these targets are real future values via shift(-N) on the hourly
# series in backfill_data.py / feature_pipeline.py -- not simulated).
# "next_hour" keeps its original unsuffixed model names for backward
# compatibility with the existing dashboard KPI card.
HORIZONS = {
    "next_hour": "aqi_target",
    "day1_24h": "aqi_target_24h",
    "day2_48h": "aqi_target_48h",
    "day3_72h": "aqi_target_72h",
}

ALGORITHMS = {
    "random_forest": dict(
        needs_scaling=False,
        factory=lambda: RandomForestRegressor(
            n_estimators=100, max_depth=20, min_samples_split=10,
            min_samples_leaf=4, random_state=42, n_jobs=-1,
        ),
    ),
    "ridge": dict(needs_scaling=True, factory=lambda: Ridge(alpha=1.0, random_state=42)),
    "gradient_boosting": dict(
        needs_scaling=False,
        factory=lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
        ),
    ),
    "lasso": dict(needs_scaling=True, factory=lambda: Lasso(alpha=0.1, random_state=42, max_iter=10000)),
}


class AQIModelTrainer:
    """
    Multi-horizon ML training pipeline for AQI prediction.
    Trains Random Forest, Ridge, Gradient Boosting, Lasso, and (optionally)
    a Neural Network -- for EACH of 4 real forecast horizons: next-hour,
    +24h, +48h, +72h. Every target is a genuine future ground-truth value,
    so predictions for the 3-day forecast come from models that actually
    learned that horizon, not from recursively chaining a 1-hour model or
    fabricating a curve.
    """

    def __init__(self, feature_group_name="aqi_predictions", version=7):
        self.feature_group_name = feature_group_name
        self.version = version
        self.models = {h: {} for h in HORIZONS}
        self.results = {h: {} for h in HORIZONS}
        self.scalers = {h: StandardScaler() for h in HORIZONS}
        self.data = {}  # horizon -> dict of X_train/X_test/y_train/y_test/scaled variants

    def connect_to_hopsworks(self):
        print("=" * 80)
        print("🚀 AQI PREDICTION - MULTI-HORIZON TRAINING PIPELINE")
        print("=" * 80)
        load_dotenv()
        print("\n--> [1/6] Connecting to Hopsworks Feature Store...")
        try:
            api_key = os.getenv("HOPSWORKS_API_KEY")
            if not api_key:
                print("    ✗ HOPSWORKS_API_KEY not found in .env file")
                return False
            self.project = hopsworks.login(
                project=os.getenv("HOPSWORKS_PROJECT", "aqi_predictor_2026"),
                host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
                port=443,
                api_key_value=api_key,
            )
            self.fs = self.project.get_feature_store()
            self.mr = self.project.get_model_registry()
            print("    ✓ Successfully connected to Hopsworks")
            return True
        except Exception as e:
            print(f"    ✗ Error connecting to Hopsworks: {e}")
            return False

    def load_features_and_targets(self):
        print("\n--> [2/6] Fetching historical features and targets...")
        try:
            print(f"    📥 Loading feature group: {self.feature_group_name} (v{self.version})")
            fg = self.fs.get_feature_group(name=self.feature_group_name, version=self.version)
            if fg is None:
                print(f"    ✗ Feature group '{self.feature_group_name}' not found!")
                return False

            try:
                self.df = fg.read()
            except Exception as flight_err:
                print(f"    ⚠ Arrow Flight read failed ({flight_err}); retrying with Hive engine...")
                self.df = fg.read(read_options={"use_hive": True})

            if self.df is None or len(self.df) == 0:
                print("    ✗ Feature group is empty!")
                return False

            print(f"    ✓ Loaded {len(self.df)} records, columns: {list(self.df.columns)}")
        except Exception as e:
            print(f"    ✗ Error loading feature group: {e}")
            import traceback
            traceback.print_exc()
            return False
        return True

    def prepare_feature_columns(self):
        """Feature columns are identical across all horizons -- only the
        target column differs -- so every target must be excluded here,
        not just the one currently being trained."""
        print("\n--> [3/6] Identifying shared feature columns...")
        metadata_cols = {'date', 'city', 'timestamp', 'datetime', 'time', 'id', 'index'}
        metadata_cols |= set(HORIZONS.values())  # exclude ALL target columns from features

        self.feature_cols = [
            c for c in self.df.columns
            if c not in metadata_cols and c.lower() not in {m.lower() for m in metadata_cols}
        ]
        if not self.feature_cols:
            print("    ✗ No feature columns found!")
            return False

        print(f"    ✓ {len(self.feature_cols)} shared features: {self.feature_cols}")
        return True

    def prepare_horizon_data(self, horizon: str, target_col: str) -> bool:
        """Build the train/test split for one specific horizon. Rows
        without a real future value yet for THIS horizon (NaN target) are
        dropped -- e.g. the last 72 hours of data can't have a 72h-ahead
        ground truth yet, so they're simply excluded from that horizon's
        training set (they're still usable for shorter horizons)."""
        if target_col not in self.df.columns:
            print(f"    ✗ Target column '{target_col}' not in feature group -- skipping {horizon}")
            return False

        X = self.df[self.feature_cols].copy()
        y = self.df[target_col].copy()

        valid_mask = ~y.isna()
        X, y = X[valid_mask], y[valid_mask]
        if X.isna().sum().sum() > 0:
            X = X.fillna(X.median())

        if len(X) < 100:
            print(f"    ✗ [{horizon}] Insufficient data ({len(X)} rows with a real target). Skipping.")
            return False

        if 'date' in self.df.columns:
            sort_cols = ['date', 'hour'] if 'hour' in self.df.columns else ['date']
            sort_idx = self.df.loc[valid_mask, sort_cols].sort_values(sort_cols).index
            X = X.loc[sort_idx].reset_index(drop=True)
            y = y.loc[sort_idx].reset_index(drop=True)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        scaler = self.scalers[horizon]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.data[horizon] = dict(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
        )
        print(f"    ✓ [{horizon}] train={len(X_train)} test={len(X_test)} (target: {target_col})")
        return True

    def _calculate_metrics(self, y_train, train_preds, y_test, test_preds):
        mask = y_test != 0
        mape = float(np.mean(np.abs((y_test[mask] - test_preds[mask]) / y_test[mask])) * 100) if mask.sum() else 0.0
        return {
            'train_mae': float(mean_absolute_error(y_train, train_preds)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_preds))),
            'train_r2': float(r2_score(y_train, train_preds)),
            'test_mae': float(mean_absolute_error(y_test, test_preds)),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_preds))),
            'test_r2': float(r2_score(y_test, test_preds)),
            'test_mape': mape,
        }

    def train_algorithm(self, horizon: str, algo_name: str):
        cfg = ALGORITHMS[algo_name]
        d = self.data[horizon]
        Xtr = d['X_train_scaled'] if cfg['needs_scaling'] else d['X_train']
        Xte = d['X_test_scaled'] if cfg['needs_scaling'] else d['X_test']

        model = cfg['factory']()
        model.fit(Xtr, d['y_train'])
        train_preds, test_preds = model.predict(Xtr), model.predict(Xte)

        self.models[horizon][algo_name] = model
        self.results[horizon][algo_name] = self._calculate_metrics(
            d['y_train'], train_preds, d['y_test'], test_preds
        )

    def train_neural_network(self, horizon: str):
        if not DEEP_LEARNING_AVAILABLE:
            return
        d = self.data[horizon]
        try:
            model = Sequential([
                Dense(128, activation='relu', input_shape=(d['X_train_scaled'].shape[1],)),
                Dropout(0.3), Dense(64, activation='relu'),
                Dropout(0.2), Dense(32, activation='relu'), Dense(1),
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
            model.fit(
                d['X_train_scaled'], d['y_train'], validation_split=0.2,
                epochs=100, batch_size=32, callbacks=[early_stop], verbose=0,
            )
            train_preds = model.predict(d['X_train_scaled'], verbose=0).flatten()
            test_preds = model.predict(d['X_test_scaled'], verbose=0).flatten()
            self.models[horizon]['neural_network'] = model
            self.results[horizon]['neural_network'] = self._calculate_metrics(
                d['y_train'], train_preds, d['y_test'], test_preds
            )
        except Exception as e:
            print(f"    ⚠️  [{horizon}] Neural Network training failed: {e}")

    def train_all_horizons(self):
        print("\n--> [4/6] Training all algorithms for every horizon...")
        for horizon, target_col in HORIZONS.items():
            print(f"\n  === Horizon: {horizon} (target: {target_col}) ===")
            if not self.prepare_horizon_data(horizon, target_col):
                continue
            for algo_name in ALGORITHMS:
                print(f"    Training {algo_name}...")
                self.train_algorithm(horizon, algo_name)
            print("    Training neural_network...")
            self.train_neural_network(horizon)
        print("\n    ✓ All horizons trained!")

    def display_results(self):
        print("\n--> [5/6] Model Performance Comparison (per horizon)...")
        for horizon, results in self.results.items():
            if not results:
                continue
            print("\n" + "=" * 90)
            print(f"📊 HORIZON: {horizon}")
            print("=" * 90)
            print(f"{'Model':<20} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<12} {'Test MAPE':<12}")
            print("-" * 90)
            for name, m in sorted(results.items(), key=lambda x: x[1]['test_rmse']):
                print(f"{name:<20} {m['test_mae']:<12.2f} {m['test_rmse']:<12.2f} {m['test_r2']:<12.4f} {m['test_mape']:<12.2f}%")
            best = min(results.items(), key=lambda x: x[1]['test_rmse'])
            print(f"🏆 Best for {horizon}: {best[0].upper()} (RMSE={best[1]['test_rmse']:.2f})")

    def register_models_to_hopsworks(self):
        print("\n--> [6/6] Registering all horizon models to Hopsworks Model Registry...")
        for horizon, algo_models in self.models.items():
            # next_hour keeps the original unsuffixed names for backward
            # compatibility with the existing dashboard; other horizons get
            # a suffix so they're distinct, separately-selectable models.
            suffix = "" if horizon == "next_hour" else f"_{horizon}"

            for algo_name, model in algo_models.items():
                model_name = f"aqi_{algo_name}_model{suffix}"
                print(f"\n    📦 Registering {model_name}...")
                try:
                    model_dir = f"aqi_models/{horizon}/{algo_name}"
                    os.makedirs(model_dir, exist_ok=True)

                    if algo_name == 'neural_network' and DEEP_LEARNING_AVAILABLE:
                        model.save(os.path.join(model_dir, "model.h5"))
                        joblib.dump(self.scalers[horizon], os.path.join(model_dir, "scaler.pkl"))
                    elif algo_name in ('ridge', 'lasso'):
                        joblib.dump(model, os.path.join(model_dir, "model.pkl"))
                        joblib.dump(self.scalers[horizon], os.path.join(model_dir, "scaler.pkl"))
                    else:
                        joblib.dump(model, os.path.join(model_dir, "model.pkl"))

                    with open(os.path.join(model_dir, "feature_names.txt"), 'w') as f:
                        f.write('\n'.join(self.feature_cols))

                    metrics = self.results[horizon][algo_name]
                    with open(os.path.join(model_dir, "metadata.txt"), 'w') as f:
                        f.write(f"Model: {algo_name}\nHorizon: {horizon}\n")
                        f.write(f"Feature Group: {self.feature_group_name} v{self.version}\n")
                        f.write(f"Test RMSE: {metrics['test_rmse']:.2f}\n")
                        f.write(f"Test R²: {metrics['test_r2']:.4f}\n")

                    d = self.data[horizon]
                    aqi_model = self.mr.python.create_model(
                        name=model_name,
                        metrics={
                            "test_mae": metrics['test_mae'], "test_rmse": metrics['test_rmse'],
                            "test_r2": metrics['test_r2'], "test_mape": metrics['test_mape'],
                            "train_r2": metrics['train_r2'],
                        },
                        input_example=d['X_train'].head(1),
                        description=(
                            f"{algo_name.replace('_', ' ').title()} for Lahore AQI, horizon={horizon}. "
                            f"RMSE={metrics['test_rmse']:.2f}, R²={metrics['test_r2']:.4f}"
                        ),
                    )
                    aqi_model.save(model_dir)
                    print(f"    ✓ {model_name} registered successfully!")
                except Exception as e:
                    print(f"    ✗ Error registering {model_name}: {e}")
                    import traceback
                    traceback.print_exc()

    def run_pipeline(self):
        if not self.connect_to_hopsworks():
            return False
        if not self.load_features_and_targets():
            return False
        if not self.prepare_feature_columns():
            return False
        self.train_all_horizons()
        self.display_results()
        self.register_models_to_hopsworks()
        print("\n" + "=" * 80)
        print("✅ MULTI-HORIZON TRAINING PIPELINE COMPLETED!")
        print("=" * 80)
        return True


def main():
    trainer = AQIModelTrainer(feature_group_name="aqi_predictions", version=7)
    success = trainer.run_pipeline()
    print("\n🎯 Done!" if success else "\n❌ Training pipeline failed.")


if __name__ == "__main__":
    main()