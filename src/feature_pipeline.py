"""
Live Feature Pipeline (runs periodically, e.g. hourly via GitHub Actions)

IMPORTANT: This now uses Open-Meteo -- the SAME source and hourly
resolution as backfill_data.py -- instead of AQICN's daily forecast feed.
Mixing an hourly-trained model with daily-resolution live data was the
root cause of the "sources don't match" issue flagged in review. Reusing
backfill_data.py's fetch/transform functions (instead of re-implementing
them) guarantees the schema can never drift apart again.

This script simply re-fetches a short recent window (enough hours to
recompute lag_1 / rolling_mean_3 correctly) and upserts it into the same
Hopsworks feature group used for training. It does NOT fabricate a
forecast -- the actual 3-day forecast is produced by the trained models
in inference_pipeline.py, using these real, latest features as input.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backfill_data import (
    fetch_open_meteo_historical_data, fetch_open_meteo_weather_data,
    merge_pollutant_and_weather, transform_and_align_schema,
)
from uploading_to_hopsworks import sanitize_integer_columns

FEATURE_GROUP_NAME = "aqi_predictions"
FEATURE_GROUP_VERSION = 7

# Enough recent hours to correctly recompute lag_1 / rolling_mean_3 for the
# newest rows. A few days of overlap is cheap and makes the upsert
# idempotent/safe to re-run.
RECENT_DAYS_WINDOW = 5


def fetch_latest_features() -> pd.DataFrame:
    print(f"--> Fetching latest {RECENT_DAYS_WINDOW}-day window from Open-Meteo (live features)...")
    raw_df = fetch_open_meteo_historical_data(days_back=RECENT_DAYS_WINDOW, days_forward=0)
    # use_archive=False: near-real-time forecast API has no reanalysis lag,
    # so it actually has data for the last few hours 
    weather_df = fetch_open_meteo_weather_data(days_back=RECENT_DAYS_WINDOW, days_forward=0, use_archive=False)
    merged_df = merge_pollutant_and_weather(raw_df, weather_df)
    df = transform_and_align_schema(merged_df)
    return sanitize_integer_columns(df)  # match the double/bigint types the FG schema expects


def upload_to_feature_store(df: pd.DataFrame) -> None:
    import hopsworks

    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY not found in .env")

    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT", "aqi_predictor_2026"),
        host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        port=443,
        api_key_value=api_key,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    if fg is None:
        raise RuntimeError(
            f"Feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION} not found. "
            "Run uploading_to_hopsworks.py once first to create it with the new schema."
        )

    print(f"--> Upserting {len(df)} rows into '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}...")
    job, _ = fg.insert(df, write_options={"wait_for_job": True})
    print("--> Upsert complete.")


def main() -> int:
    print("=" * 60)
    print("LIVE FEATURE PIPELINE (Open-Meteo, hourly -- matches training source)")
    print("=" * 60)
    try:
        df = fetch_latest_features()
        upload_to_feature_store(df)
        print(f"\n[SUCCESS] {len(df)} rows refreshed in the feature store.")
        return 0
    except Exception as e:
        print(f"\n[FAILURE] Live feature pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())