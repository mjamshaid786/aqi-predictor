import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "aqi_features_lahore.csv"

# Lahore Location Parameters
LATITUDE = 31.5204
LONGITUDE = 74.3587
CITY_NAME = "lahore"

WEATHER_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,wind_speed_10m,"
    "wind_direction_10m,surface_pressure,precipitation,cloud_cover"
)


def fetch_open_meteo_weather_data(days_back: int, days_forward: int = 0, use_archive: bool = True) -> pd.DataFrame:
    """
    Fetch hourly WEATHER data (temperature, humidity, wind, pressure, etc.)
    -- separate from the pollutant data above. Current weather is a strong
    predictor of how pollution will disperse/build up over the next few
    days (e.g. wind speed/direction), which the pollutant-only features
    were missing entirely.

    use_archive=True: archive-api.open-meteo.com (deep historical
    reanalysis, good for the 365-day backfill, but has a few days' lag
    before the very latest data is available).
    use_archive=False: api.open-meteo.com/v1/forecast with past_days
    (near-real-time, no lag -- used by feature_pipeline.py for the small
    recent window).
    """
    end_date = (datetime.now() + timedelta(days=days_forward)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    if use_archive:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "start_date": start_date, "end_date": end_date,
            "hourly": WEATHER_HOURLY_VARS, "timezone": "auto",
        }
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "past_days": days_back, "forecast_days": max(days_forward, 1),
            "hourly": WEATHER_HOURLY_VARS, "timezone": "auto",
        }

    print(f"--> Fetching weather data from Open-Meteo ({'archive' if use_archive else 'forecast'} API)...")
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        hourly_data = response.json().get("hourly", {})
        if not hourly_data:
            raise RuntimeError("Weather API returned empty hourly data block.")
        df = pd.DataFrame(hourly_data).rename(columns={"time": "date"})
        print(f"--> Successfully fetched {len(df)} hourly weather records.")
        return df
    except Exception as e:
        raise RuntimeError(f"Open-Meteo weather API call failed: {e}")


def merge_pollutant_and_weather(pollutant_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge on the shared hourly timestamp column."""
    pollutant_df = pollutant_df.copy()
    weather_df = weather_df.copy()
    pollutant_df["date"] = pd.to_datetime(pollutant_df["date"])
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    merged = pd.merge(pollutant_df, weather_df, on="date", how="left")
    return merged

def fetch_open_meteo_historical_data(days_back: int = 90, days_forward: int = 0) -> pd.DataFrame:
    """
    Fetch hourly air-quality data from Open-Meteo
    days_back: how many past days of ACTUAL observed data to include.
    days_forward: how many days of Open-Meteo's own forecast to include
                  (0 for pure backfill; feature_pipeline.py uses a small
                  days_back window + days_forward=0 too, since our own
                  trained models -- not Open-Meteo's forecast -- are what
                  produce the AQI forecast).
    """
    label = f"{days_back}-day" if days_forward == 0 else f"{days_back}-day + {days_forward}-day forecast"
    print(f"--> Fetching {label} data from Open-Meteo API...")

    end_date = (datetime.now() + timedelta(days=days_forward)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        hourly_data = data.get("hourly", {})
        if not hourly_data:
            raise RuntimeError("API returned empty hourly data block.")
            
        df = pd.DataFrame(hourly_data)
        df = df.rename(columns={"time": "date"})
        print(f"--> Successfully fetched {len(df)} hourly records from API.")
        return df
        
    except Exception as e:
        raise RuntimeError(f"Open-Meteo API call failed: {e}")

def transform_and_align_schema(df_om: pd.DataFrame) -> pd.DataFrame:
    print("--> [2/3] Aligning 100% exact columns and handling NaN for Bigint casting...")
    
    # Date indexing parsing
    df_om['date'] = pd.to_datetime(df_om['date'])
    
    df_transformed = pd.DataFrame()
    
    # 1. Primary Keys & String Definitions
    df_transformed['date'] = df_om['date'].dt.strftime('%Y-%m-%d').astype(str)
    df_transformed['city'] = CITY_NAME
    
    # Timestamp column
    # BUG FIX: this previously called datetime.now(timezone.utc) once and
    # assigned that SAME value to every row, which made every hourly record
    # collapse onto an identical timestamp -- and therefore an identical
    # (city, date, hour) primary key in Hopsworks, wiping out all but one
    # row on upsert. The real per-row observation time is already sitting
    # in df_om['date'] (that's what 'hour'/'day'/'month' below are derived
    # from) -- so 'timestamp' must be derived from THAT, not from "now".
    df_transformed['timestamp'] = df_om['date']
    
    # Helper function taake integer casting mein NaN ka error kabhi na aaye
    def safe_int_cast(series):
        return pd.to_numeric(series, errors='coerce').fillna(0).round().astype('int64')
        
    # Helper function for float casting
    def safe_float_cast(series):
        return pd.to_numeric(series, errors='coerce').fillna(0.0).astype('float64')
    
    # 2. DOUBLE (float64) Columns
    df_transformed['o3_avg'] = safe_float_cast(df_om['ozone'])
    df_transformed['o3_max'] = safe_float_cast(df_om['ozone'])
    df_transformed['pm10_min'] = safe_float_cast(df_om['pm10'])
    df_transformed['pm25_max'] = safe_float_cast(df_om['pm2_5'])
    
    df_transformed['uvi_avg'] = pd.Series([0.0] * len(df_om), dtype='float64')
    df_transformed['uvi_max'] = pd.Series([0.0] * len(df_om), dtype='float64')
    
    # 3. BIGINT (int64) Columns - Strictly handling NaNs here
    df_transformed['pm10_avg'] = safe_int_cast(df_om['pm10'])
    df_transformed['pm10_max'] = safe_int_cast(df_om['pm10'])
    df_transformed['pm25_avg'] = safe_int_cast(df_om['pm2_5'])
    df_transformed['pm25_min'] = safe_int_cast(df_om['pm2_5'])
    
    df_transformed['hour'] = safe_int_cast(df_om['date'].dt.hour)
    df_transformed['day'] = safe_int_cast(df_om['date'].dt.day)
    df_transformed['month'] = safe_int_cast(df_om['date'].dt.month)
    df_transformed['day_of_week'] = safe_int_cast(df_om['date'].dt.dayofweek)
    df_transformed['is_weekend'] = safe_int_cast(df_transformed['day_of_week'].isin([5, 6]))

    # Weather features (current atmospheric state) -- these are the main
    # signal that was missing for real multi-day-ahead forecasting: wind
    # speed/direction affects how fast pollution disperses, pressure
    # systems and precipitation strongly affect pollutant buildup over the
    # following days. If the weather columns weren't merged in (e.g. the
    # weather API call failed upstream), fill with 0 rather than crashing.
    weather_cols = [
        'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
        'wind_direction_10m', 'surface_pressure', 'precipitation', 'cloud_cover',
    ]
    for col in weather_cols:
        df_transformed[col] = safe_float_cast(df_om[col]) if col in df_om.columns else 0.0
    
    # Sorting & Shifting Operations
    df_transformed = df_transformed.sort_values(['city', 'date', 'hour']).reset_index(drop=True)
    grouped = df_transformed.groupby('city', group_keys=False)
    
    # 4. Calculated Sequence Metrics (DOUBLE)
    df_transformed['aqi_lag_1'] = grouped['pm25_avg'].shift(1)
    df_transformed['aqi_lag_1'] = safe_float_cast(df_transformed['aqi_lag_1'])
    
    # Prevent divide by zero scenarios
    denom = df_transformed['aqi_lag_1'].replace(0, np.nan)
    df_transformed['aqi_change_rate'] = ((df_transformed['pm25_avg'] - df_transformed['aqi_lag_1']) / denom) * 100.0
    df_transformed['aqi_change_rate'] = safe_float_cast(df_transformed['aqi_change_rate'])
    
    df_transformed['aqi_rolling_mean_3'] = grouped['pm25_avg'].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    df_transformed['aqi_rolling_mean_3'] = safe_float_cast(df_transformed['aqi_rolling_mean_3'])
    
    df_transformed['aqi_target'] = grouped['pm25_avg'].shift(-1)          # +1 hour (existing next-hour target)
    df_transformed['aqi_target'] = safe_float_cast(df_transformed['aqi_target'])

    # Direct multi-horizon targets for the 3-day forecast — real future
    # ground-truth values (shift on the actual hourly series), NOT a
    # simulated/random projection. Rows near the end of the dataset won't
    # have these yet (NaN) since the future hasn't happened -- that's
    # expected and those rows are simply excluded from training for that
    # horizon (handled in training_pipeline.py).
    df_transformed['aqi_target_24h'] = safe_float_cast(grouped['pm25_avg'].shift(-24))
    df_transformed['aqi_target_48h'] = safe_float_cast(grouped['pm25_avg'].shift(-48))
    df_transformed['aqi_target_72h'] = safe_float_cast(grouped['pm25_avg'].shift(-72))
    
    # 100% Exact Sequence Ordering matching your Hopsworks schema
    hopsworks_exact_columns = [
        'timestamp', 'o3_avg', 'o3_max', 'pm10_avg', 'pm10_max', 'pm10_min',
        'pm25_avg', 'pm25_max', 'pm25_min', 'uvi_avg', 'uvi_max', 'city',
        'hour', 'day', 'month', 'day_of_week', 'is_weekend',
        'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
        'wind_direction_10m', 'surface_pressure', 'precipitation', 'cloud_cover',
        'aqi_lag_1', 'aqi_change_rate', 'aqi_rolling_mean_3',
        'aqi_target', 'aqi_target_24h', 'aqi_target_48h', 'aqi_target_72h', 'date'
    ]
    
    return df_transformed[hopsworks_exact_columns]
if __name__ == "__main__":
    print("=" * 60)
    print("AUTOMATED AQI BACKFILL DATA PIPELINE (RAW STORE)")
    print("=" * 60)
    
    start_time = time.perf_counter()
    
    try:
        # Try progressively smaller windows -- Open-Meteo's exact historical
        # limit for this air-quality dataset isn't documented, so instead
        # of guessing and hard-failing, request the largest useful window
        # first and fall back automatically if the API can't serve it.
        for DAYS_BACK in (730, 500, 365):
            try:
                print(f"\n--> Attempting {DAYS_BACK}-day backfill...")
                raw_api_df = fetch_open_meteo_historical_data(days_back=DAYS_BACK, days_forward=0)
                weather_df = fetch_open_meteo_weather_data(days_back=DAYS_BACK, days_forward=0, use_archive=True)
                break
            except Exception as e:
                print(f"    ⚠ {DAYS_BACK}-day request failed ({e}); trying a smaller window...")
        else:
            raise RuntimeError("Could not fetch backfill data at any tested window size (730/500/365 days).")

        merged_df = merge_pollutant_and_weather(raw_api_df, weather_df)
        actual_days = (pd.to_datetime(merged_df['date']).max() - pd.to_datetime(merged_df['date']).min()).days
        print(f"--> Actual data span received: {actual_days} days ({len(merged_df)} hourly rows)")
        
        # Step 2: Schema Processing
        final_df = transform_and_align_schema(merged_df)
        
        # Step 3: Write matrix
        print(f"--> [3/3] Saving final matrix to: {OUTPUT_CSV}")
        final_df.to_csv(OUTPUT_CSV, index=False)
        
        print("\n" + "-"*40)
        print("PIPELINE EXECUTION METRICS:")
        print(f"Total Backfilled Rows: {final_df.shape[0]}")
        print(f"Total Aligned Columns: {final_df.shape[1]}")
        print(f"Execution Duration:    {time.perf_counter() - start_time:.2f} seconds")
        print("-"*40)
        
        print("\nSAMPLE REGISTERED MATRIX VIEW (First 2 Hourly Rows):")
        print(final_df[['date', 'city', 'pm25_avg', 'hour', 'aqi_target']].head(2))
        print("=" * 60)
        print("\n[SUCCESS] Jamshaid bhai, bina filter kiye poora data file mein store ho gaya hai!")
        
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Pipeline execution halted: {e}")
        sys.exit(1)