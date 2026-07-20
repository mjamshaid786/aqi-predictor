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

def fetch_open_meteo_historical_data() -> pd.DataFrame:
    print("--> [1/3] Fetching 90-day historical data from Open-Meteo API...")
    
    # Calculate start and end dates for past 90 days
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
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
    
    df_transformed['aqi_target'] = grouped['pm25_avg'].shift(-1)
    df_transformed['aqi_target'] = safe_float_cast(df_transformed['aqi_target'])
    
    # 100% Exact Sequence Ordering matching your Hopsworks schema
    hopsworks_exact_columns = [
        'timestamp', 'o3_avg', 'o3_max', 'pm10_avg', 'pm10_max', 'pm10_min',
        'pm25_avg', 'pm25_max', 'pm25_min', 'uvi_avg', 'uvi_max', 'city',
        'hour', 'day', 'month', 'day_of_week', 'is_weekend',
        'aqi_lag_1', 'aqi_change_rate', 'aqi_rolling_mean_3', 'aqi_target', 'date'
    ]
    
    return df_transformed[hopsworks_exact_columns]
if __name__ == "__main__":
    print("=" * 60)
    print("AUTOMATED AQI BACKFILL DATA PIPELINE (RAW STORE)")
    print("=" * 60)
    
    start_time = time.perf_counter()
    
    try:
        # Step 1: API Request
        raw_api_df = fetch_open_meteo_historical_data()
        
        # Step 2: Schema Processing
        final_df = transform_and_align_schema(raw_api_df)
        
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