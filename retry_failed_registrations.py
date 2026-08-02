"""
Retry registration for just the models that failed last run, reusing the
already-trained local model files (no retraining needed).
Run standalone: python retry_failed_registrations.py
"""
import os
from pathlib import Path
import hopsworks
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

# (model_name, local_model_dir, metrics) -- metrics copied from the
# training run's printed comparison table for these exact 3 models.
FAILED_MODELS = [
    (
        "aqi_neural_network_model_day1_24h",
        "aqi_models/day1_24h/neural_network",
        {"test_mae": 20.65, "test_rmse": 28.66, "test_r2": 0.0263, "test_mape": 38.51},
    ),
    (
        "aqi_ridge_model_day2_48h",
        "aqi_models/day2_48h/ridge",
        {"test_mae": 25.52, "test_rmse": 35.54, "test_r2": -0.4410, "test_mape": 44.15},
    ),
    (
        "aqi_neural_network_model_day2_48h",
        "aqi_models/day2_48h/neural_network",
        {"test_mae": 24.77, "test_rmse": 33.61, "test_r2": -0.2890, "test_mape": 42.76},
    ),
]


def main():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT", "aqi_predictor_2026"),
        host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        port=443,
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    mr = project.get_model_registry()

    for model_name, model_dir, metrics in FAILED_MODELS:
        full_dir = PROJECT_ROOT / model_dir
        if not full_dir.exists():
            print(f"\n✗ SKIP {model_name}: local dir not found at {full_dir} "
                  f"(re-run training_pipeline.py to regenerate it)")
            continue

        print(f"\n📦 Retrying {model_name} (from {full_dir})...")
        try:
            aqi_model = mr.python.create_model(
                name=model_name,
                metrics=metrics,
                description=f"Retried registration for {model_name}. "
                             f"RMSE={metrics['test_rmse']:.2f}, R²={metrics['test_r2']:.4f}",
            )
            aqi_model.save(str(full_dir))
            print(f"✓ {model_name} registered successfully!")
        except Exception as e:
            print(f"✗ {model_name} failed again: {e}")


if __name__ == "__main__":
    main()