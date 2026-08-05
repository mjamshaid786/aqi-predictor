"""
Real inference logic for the AQI dashboard's 3-day forecast.

Each horizon (day1=+24h, day2=+48h, day3=+72h) w trained as a DIRECT
target (real future ground-truth value via shift(-24/-48/-72) on the
hourly series -- see backfill_data.py). So producing the 3-day forecast
means: take the model trained for that specific horizon, feed it the
LATEST real feature row (i.e. "now"), and read off its prediction. No
recursive chaining, no random walk, no fabricated curve.
"""
import os
import joblib
import numpy as np
import pandas as pd

ALGO_BASE_NAMES = ["neural_network", "lasso", "gradient_boosting", "ridge", "random_forest"]

# horizon key -> (model name suffix, hours ahead, display label)
FORECAST_HORIZONS = {
    "day1_24h": ("_day1_24h", 24, "Day 1 (+24h)"),
    "day2_48h": ("_day2_48h", 48, "Day 2 (+48h)"),
    "day3_72h": ("_day3_72h", 72, "Day 3 (+72h)"),
}


def get_project():
    import hopsworks
    return hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT", "aqi_predictor_2026"),
        host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        port=443,
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )


def load_best_model(mr, candidate_names):
    """Shared loader: try each candidate model name, best-to-worst by
    test_rmse, skipping any whose registry entry has no usable weights
    (handles the earlier corrupted-registration issue gracefully)."""
    candidates = []
    for name in candidate_names:
        try:
            candidates.extend(mr.get_models(name))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError(f"None of these models exist in the registry: {candidate_names}")

    def rmse(m):
        return m.training_metrics.get("test_rmse", float("inf")) if m.training_metrics else float("inf")

    max_version = max(int(m.version) for m in candidates)
    ordered = sorted([m for m in candidates if int(m.version) == max_version], key=rmse)

    errors = []
    for model_meta in ordered:
        try:
            model_dir = model_meta.download()
        except Exception as e:
            errors.append(f"{model_meta.name} v{model_meta.version}: download failed ({e})")
            continue

        loaded = None
        for fname in ("model.pkl", "model.joblib"):
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                loaded = joblib.load(fpath)
                break
        if loaded is None:
            h5_path = os.path.join(model_dir, "model.h5")
            if os.path.exists(h5_path):
                try:
                    from tensorflow.keras.models import load_model as keras_load_model
                    loaded = keras_load_model(h5_path, compile=False)
                except Exception as e:
                    errors.append(f"{model_meta.name} v{model_meta.version}: model.h5 load failed ({e})")
                    continue

        if loaded is not None:
            feature_names = None
            feat_path = os.path.join(model_dir, "feature_names.txt")
            if os.path.exists(feat_path):
                with open(feat_path) as f:
                    feature_names = [line.strip() for line in f if line.strip()]
            scaler = None
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            return dict(
                model=loaded, scaler=scaler, feature_names=feature_names,
                name=model_meta.name, version=model_meta.version,
                metrics=model_meta.training_metrics,
            )
        errors.append(f"{model_meta.name} v{model_meta.version}: no usable weights file")

    raise RuntimeError("No usable model found:\n" + "\n".join(errors))


def load_model_for_horizon(mr, horizon_key: str):
    """horizon_key: 'next_hour', 'day1_24h', 'day2_48h', or 'day3_72h'."""
    suffix = "" if horizon_key == "next_hour" else FORECAST_HORIZONS[horizon_key][0]
    candidate_names = [f"aqi_{algo}_model{suffix}" for algo in ALGO_BASE_NAMES]
    return load_best_model(mr, candidate_names)


def predict_from_row(model_bundle: dict, row: pd.Series) -> float:
    """Predict a single value from one feature row using a loaded model bundle."""
    feature_names = model_bundle["feature_names"]
    X = pd.DataFrame([row[feature_names]]) if feature_names else pd.DataFrame([row])
    if model_bundle["scaler"] is not None:
        X = model_bundle["scaler"].transform(X)
    pred = np.asarray(model_bundle["model"].predict(X)).reshape(-1)[0]
    return float(pred)


def generate_3day_forecast(df: pd.DataFrame, mr) -> pd.DataFrame:
    """
    Real 3-day forecast: for each horizon, load that horizon's best model
    and predict from the LATEST actual feature row. Returns one real point
    per horizon (not a fabricated 72-point curve) -- that matches exactly
    what these direct-horizon models were trained to produce.
    """
    latest_row = df.iloc[-1]
    latest_time = pd.to_datetime(latest_row.get("timestamp", pd.Timestamp.now()))

    rows = []
    for horizon_key, (_, hours_ahead, label) in FORECAST_HORIZONS.items():
        try:
            bundle = load_model_for_horizon(mr, horizon_key)
            predicted = predict_from_row(bundle, latest_row)
            rows.append(dict(
                horizon=horizon_key, label=label,
                forecast_time=latest_time + pd.Timedelta(hours=hours_ahead),
                predicted_aqi=predicted,
                model_name=bundle["name"], model_version=bundle["version"],
            ))
        except Exception as e:
            rows.append(dict(
                horizon=horizon_key, label=label,
                forecast_time=latest_time + pd.Timedelta(hours=hours_ahead),
                predicted_aqi=None, model_name=None, model_version=None, error=str(e),
            ))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    project = get_project()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    fg = fs.get_feature_group("aqi_predictions", version=7)
    df = fg.read().sort_values("timestamp").reset_index(drop=True)

    forecast = generate_3day_forecast(df, mr)
    print("\n3-Day Forecast (real model predictions):")
    print(forecast[["label", "forecast_time", "predicted_aqi", "model_name"]].to_string(index=False))