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

# Optional: Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.callbacks import EarlyStopping
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not installed. Skipping deep learning models.")
    DEEP_LEARNING_AVAILABLE = False


class AQIModelTrainer:
    """
    Complete ML Training Pipeline for AQI Prediction
    Trains multiple models as per PDF requirements:
    - Random Forest
    - Ridge Regression
    - Gradient Boosting
    - Lasso Regression
    - Neural Network (optional)
    """
    
    def __init__(self, feature_group_name="aqi_predictions", version=6):
        self.feature_group_name = feature_group_name
        self.version = version
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def connect_to_hopsworks(self):
        """Step 1: Connect to Hopsworks Feature Store"""
        print("=" * 80)
        print("🚀 AQI PREDICTION - MULTI-MODEL TRAINING PIPELINE")
        print("=" * 80)
        
        load_dotenv()
        
        print("\n--> [1/6] Connecting to Hopsworks Feature Store...")
        try:
            api_key = os.getenv("HOPSWORKS_API_KEY")
            if not api_key:
                print("    ✗ HOPSWORKS_API_KEY not found in .env file")
                return False
            self.project = hopsworks.login(api_key_value=api_key)
            self.fs = self.project.get_feature_store()
            self.mr = self.project.get_model_registry()
            print("    ✓ Successfully connected to Hopsworks")
            return True
        except Exception as e:
            print(f"    ✗ Error connecting to Hopsworks: {e}")
            return False
    
    def load_features_and_targets(self):
        """Step 2: Fetch historical (features, targets) from Feature Store"""
        print("\n--> [2/6] Fetching historical features and targets...")
        
        try:
            print(f"    📥 Loading feature group: {self.feature_group_name} (v{self.version})")
            
            fg = self.fs.get_feature_group(
                name=self.feature_group_name, 
                version=self.version
            )
            
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
            
            print(f"    ✓ Loaded {len(self.df)} records")
            print(f"    ✓ Columns: {list(self.df.columns)}")
            
            # Display column info
            print(f"    ✓ Shape: {self.df.shape}")
            
        except Exception as e:
            print(f"    ✗ Error loading feature group: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def identify_target_column(self):
        """Identify the target column (AQI value to predict)"""
        print("\n--> [3/6] Identifying features and target...")
        
        # Common AQI target column names
        possible_targets = [
            'aqi_target', 'aqi', 'pm25', 'pm2.5', 
            'target', 'aqi_value', 'air_quality_index'
        ]
        
        target_col = None
        for col in possible_targets:
            if col in self.df.columns:
                target_col = col
                break
        
        if target_col is None:
            # Try to find numeric column that looks like AQI
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            print(f"    Available numeric columns: {list(numeric_cols)}")
            
            # If there's only one numeric column, use it
            if len(numeric_cols) == 1:
                target_col = numeric_cols[0]
                print(f"    ✓ Auto-selected target: {target_col}")
            else:
                print("    ✗ Cannot identify target column automatically")
                print("    Available columns:", list(self.df.columns))
                return None
        else:
            print(f"    ✓ Target column: {target_col}")
        
        return target_col
    
    def prepare_data(self):
        """Prepare features and targets"""
        
        # Identify target column
        target_col = self.identify_target_column()
        
        if target_col is None:
            return False
        
        # Define metadata columns to exclude from features
        metadata_cols = [
            'date', 'city', 'timestamp', target_col,
            'datetime', 'time', 'id', 'index'
        ]
        
        self.feature_cols = [
            col for col in self.df.columns 
            if col not in metadata_cols and col.lower() not in [m.lower() for m in metadata_cols]
        ]
        
        if len(self.feature_cols) == 0:
            print("    ✗ No feature columns found!")
            print("    All columns:", list(self.df.columns))
            return False
        
        # Separate features and target
        X = self.df[self.feature_cols].copy()
        y = self.df[target_col].copy()
        
        print(f"    ✓ Number of features: {len(self.feature_cols)}")
        print(f"    ✓ Features used:")
        for i, col in enumerate(self.feature_cols, 1):
            print(f"       {i}. {col}")
        
        # Remove rows with missing targets
        missing_targets = y.isna().sum()
        if missing_targets > 0:
            print(f"    ⚠ Removing {missing_targets} rows with missing targets")
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Handle missing values in features
        missing_count = X.isna().sum().sum()
        if missing_count > 0:
            print(f"    ⚠ Found {missing_count} missing feature values - filling with median...")
            X = X.fillna(X.median())
        
        # Check for sufficient data
        if len(X) < 100:
            print(f"    ✗ Insufficient data ({len(X)} records). Need at least 100.")
            return False
        
        # Sort by time if date column exists
        if 'date' in self.df.columns:
            if 'hour' in self.df.columns:
                sort_cols = ['date', 'hour']
            else:
                sort_cols = ['date']
            
            sort_indices = self.df[sort_cols].sort_values(sort_cols).index
            X = X.loc[sort_indices].reset_index(drop=True)
            y = y.loc[sort_indices].reset_index(drop=True)
            print(f"    ✓ Data sorted by: {sort_cols}")
        
        # Time-based train/test split (80/20)
        split_idx = int(len(X) * 0.8)
        
        self.X_train, self.X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"    ✓ Train set: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"    ✓ Test set:  {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return True
    
    def train_random_forest(self):
        """Train Random Forest Regressor"""
        print("\n    🌲 Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        
        # Evaluate
        train_preds = model.predict(self.X_train)
        test_preds = model.predict(self.X_test)
        
        self.results['random_forest'] = self._calculate_metrics(
            self.y_train, train_preds, self.y_test, test_preds
        )
        
        print("    ✓ Random Forest training completed")
    
    def train_ridge_regression(self):
        """Train Ridge Regression"""
        print("    📊 Training Ridge Regression...")
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        self.models['ridge'] = model
        
        # Evaluate
        train_preds = model.predict(self.X_train_scaled)
        test_preds = model.predict(self.X_test_scaled)
        
        self.results['ridge'] = self._calculate_metrics(
            self.y_train, train_preds, self.y_test, test_preds
        )
        
        print("    ✓ Ridge Regression training completed")
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting Regressor"""
        print("    🚀 Training Gradient Boosting...")
        
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['gradient_boosting'] = model
        
        # Evaluate
        train_preds = model.predict(self.X_train)
        test_preds = model.predict(self.X_test)
        
        self.results['gradient_boosting'] = self._calculate_metrics(
            self.y_train, train_preds, self.y_test, test_preds
        )
        
        print("    ✓ Gradient Boosting training completed")
    
    def train_lasso_regression(self):
        """Train Lasso Regression"""
        print("    📉 Training Lasso Regression...")
        
        model = Lasso(alpha=0.1, random_state=42, max_iter=10000)
        model.fit(self.X_train_scaled, self.y_train)
        self.models['lasso'] = model
        
        # Evaluate
        train_preds = model.predict(self.X_train_scaled)
        test_preds = model.predict(self.X_test_scaled)
        
        self.results['lasso'] = self._calculate_metrics(
            self.y_train, train_preds, self.y_test, test_preds
        )
        
        print("    ✓ Lasso Regression training completed")
    
    def train_neural_network(self):
        """Train Deep Learning Model (Optional)"""
        if not DEEP_LEARNING_AVAILABLE:
            print("    ⚠️  Skipping Neural Network (TensorFlow not available)")
            return
        
        print("    🧠 Training Neural Network (Deep Learning)...")
        
        try:
            # Build feedforward neural network
            model = Sequential([
                Dense(128, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )
            
            # Train
            model.fit(
                self.X_train_scaled, self.y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            self.models['neural_network'] = model
            
            # Evaluate
            train_preds = model.predict(self.X_train_scaled, verbose=0).flatten()
            test_preds = model.predict(self.X_test_scaled, verbose=0).flatten()
            
            self.results['neural_network'] = self._calculate_metrics(
                self.y_train, train_preds, self.y_test, test_preds
            )
            
            print("    ✓ Neural Network training completed")
            
        except Exception as e:
            print(f"    ⚠️  Neural Network training failed: {e}")
    
    def _calculate_metrics(self, y_train, train_preds, y_test, test_preds):
        """Calculate comprehensive metrics"""
        # Prevent division by zero in MAPE
        mask = y_test != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_test[mask] - test_preds[mask]) / y_test[mask])) * 100)
        else:
            mape = 0.0
        
        return {
            'train_mae': float(mean_absolute_error(y_train, train_preds)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_preds))),
            'train_r2': float(r2_score(y_train, train_preds)),
            'test_mae': float(mean_absolute_error(y_test, test_preds)),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_preds))),
            'test_r2': float(r2_score(y_test, test_preds)),
            'test_mape': mape
        }
    
    def train_all_models(self):
        """Step 4: Train all models"""
        print("\n--> [4/6] Training multiple ML models...")
        print("    Models: Random Forest, Ridge, Gradient Boosting, Lasso, Neural Network")
        
        self.train_random_forest()
        self.train_ridge_regression()
        self.train_gradient_boosting()
        self.train_lasso_regression()
        self.train_neural_network()
        
        print("\n    ✓ All models trained successfully!")
    
    def display_results(self):
        """Step 5: Display comparison of all models"""
        print("\n--> [5/6] Model Performance Comparison...")
        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("=" * 100)
        print(f"{'Model':<25} {'Test MAE':<15} {'Test RMSE':<15} {'Test R²':<15} {'Test MAPE':<15}")
        print("-" * 100)
        
        for model_name, metrics in sorted(self.results.items(), key=lambda x: x[1]['test_rmse']):
            print(f"{model_name:<25} "
                  f"{metrics['test_mae']:<15.2f} "
                  f"{metrics['test_rmse']:<15.2f} "
                  f"{metrics['test_r2']:<15.4f} "
                  f"{metrics['test_mape']:<15.2f}%")
        
        print("=" * 100)
        
        # Find best model
        if self.results:
            best_model_name = min(self.results.items(), key=lambda x: x[1]['test_rmse'])[0]
            best_rmse = self.results[best_model_name]['test_rmse']
            best_r2 = self.results[best_model_name]['test_r2']
            
            print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
            print(f"   Test RMSE: {best_rmse:.2f}")
            print(f"   Test R²: {best_r2:.4f}")
        
        print("\n" + "=" * 100)
    
    def register_models_to_hopsworks(self):
        """Step 6: Register all trained models to Hopsworks Model Registry"""
        print("\n--> [6/6] Registering models to Hopsworks Model Registry...")
        
        for model_name, model in self.models.items():
            print(f"\n    📦 Registering {model_name}...")
            
            try:
                # Create directory for model artifacts
                model_dir = f"aqi_models/{model_name}"
                os.makedirs(model_dir, exist_ok=True)
                
                # Save model
                if model_name == 'neural_network' and DEEP_LEARNING_AVAILABLE:
                    model_path = os.path.join(model_dir, "model.h5")
                    model.save(model_path)
                    # Neural network was also trained on X_train_scaled — save the scaler too
                    joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
                elif model_name in ['ridge', 'lasso']:
                    # Save model and scaler
                    model_path = os.path.join(model_dir, "model.pkl")
                    scaler_path = os.path.join(model_dir, "scaler.pkl")
                    joblib.dump(model, model_path)
                    joblib.dump(self.scaler, scaler_path)
                else:
                    model_path = os.path.join(model_dir, "model.pkl")
                    joblib.dump(model, model_path)
                
                # Save feature names
                with open(os.path.join(model_dir, "feature_names.txt"), 'w') as f:
                    f.write('\n'.join(self.feature_cols))
                
                # Save metadata
                metrics = self.results[model_name]
                with open(os.path.join(model_dir, "metadata.txt"), 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Feature Group: {self.feature_group_name} v{self.version}\n")
                    f.write(f"Training Samples: {len(self.X_train)}\n")
                    f.write(f"Test Samples: {len(self.X_test)}\n")
                    f.write(f"Number of Features: {len(self.feature_cols)}\n")
                    f.write(f"Test RMSE: {metrics['test_rmse']:.2f}\n")
                    f.write(f"Test MAE: {metrics['test_mae']:.2f}\n")
                    f.write(f"Test R²: {metrics['test_r2']:.4f}\n")
                    f.write(f"Test MAPE: {metrics['test_mape']:.2f}%\n")
                
                # Register to Hopsworks
                input_example = self.X_train.head(1)
                
                aqi_model = self.mr.python.create_model(
                    name=f"aqi_{model_name}_model",
                    metrics={
                        "test_mae": metrics['test_mae'],
                        "test_rmse": metrics['test_rmse'],
                        "test_r2": metrics['test_r2'],
                        "test_mape": metrics['test_mape'],
                        "train_r2": metrics['train_r2']
                    },
                    input_example=input_example,
                    description=(
                        f"{model_name.replace('_', ' ').title()} for Lahore AQI prediction. "
                        f"Trained on {len(self.X_train)} samples with {len(self.feature_cols)} features. "
                        f"Performance: RMSE={metrics['test_rmse']:.2f}, "
                        f"R²={metrics['test_r2']:.4f}, "
                        f"MAE={metrics['test_mae']:.2f}"
                    )
                )
                
                aqi_model.save(model_dir)
                print(f"    ✓ {model_name} registered successfully!")
                
            except Exception as e:
                print(f"    ✗ Error registering {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    def run_pipeline(self):
        """Execute complete training pipeline"""
        if not self.connect_to_hopsworks():
            return False
        
        if not self.load_features_and_targets():
            return False
        
        if not self.prepare_data():
            return False
        
        self.train_all_models()
        self.display_results()
        self.register_models_to_hopsworks()
        
        print("\n" + "=" * 80)
        print("✅ MULTI-MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        if self.results:
            best_model = min(self.results.items(), key=lambda x: x[1]['test_rmse'])[0]
            print(f"\n📊 Summary:")
            print(f"  • Feature Group: {self.feature_group_name} v{self.version}")
            print(f"  • Total models trained: {len(self.models)}")
            print(f"  • Models registered to Hopsworks Model Registry")
            print(f"  • All models evaluated with RMSE, MAE, R², MAPE")
            print(f"  • Best performing model: {best_model}")
            print("=" * 80)
        
        return True


def main():
    """Main entry point"""
    # Use the correct feature group name and version
    trainer = AQIModelTrainer(
        feature_group_name="aqi_predictions",
        version=6
    )
    
    success = trainer.run_pipeline()
    
    if success:
        print("\n🎯 Congratulations !")
        
    else:
        print("\n❌ Training pipeline failed.")
        print("\n💡 Check the error messages above for details.")


if __name__ == "__main__":
    main()