"""
Cleaning pipeline for Lahore AQI features dataset.
Reads:  src/data/aqi_features_lahore.csv
Writes: src/data/cleaned_data.csv  (mode="w" -> always overwritten fresh)
"""

import pandas as pd
import numpy as np

INPUT_FILE = "src/data/aqi_features_lahore.csv"
OUTPUT_FILE = "src/data/cleaned_data.csv"

log = []  # processing log only, never the raw data itself


def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    log.append(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def fix_dtypes(df):
    # Datetime columns
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # The raw file may call this column "ingested_at" or already "timestamp"
    ts_col = "ingested_at" if "ingested_at" in df.columns else (
        "timestamp" if "timestamp" in df.columns else None
    )
    if ts_col is not None:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    # Numeric columns (everything except known non-numeric / datetime)
    non_numeric = {"date", "ingested_at", "timestamp", "city", "dominant_pollutant"}
    numeric_cols = [c for c in df.columns if c not in non_numeric]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Categorical columns
    for c in ["city", "dominant_pollutant"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    log.append("Fixed data types (datetime, numeric, categorical).")

    # Standardize the ingestion timestamp column name to "timestamp"
    if ts_col == "ingested_at":
        df = df.rename(columns={"ingested_at": "timestamp"})
        log.append("Renamed 'ingested_at' -> 'timestamp'.")
    elif ts_col == "timestamp":
        log.append("Found existing 'timestamp' column; parsed as datetime.")
    else:
        log.append("Warning: no 'ingested_at' or 'timestamp' column found in source data.")

    return df


def drop_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates()
    # a duplicate reading for the same city/date/hour is also a duplicate record
    subset_cols = [c for c in ["city", "date", "hour"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols, keep="first")
    after = df.shape[0]
    log.append(f"Removed {before - after} duplicate row(s) (dedup key: {subset_cols if subset_cols else 'full row'}).")
    return df


def handle_missing(df):
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Time-ordered numeric gaps: linear interpolation, then edge fill
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))

    # Categorical / object gaps: mode fill
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        if df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")

    remaining_na = int(df.isna().sum().sum())
    log.append(f"Missing values imputed. Remaining NaNs: {remaining_na}.")
    return df


def cap_outliers(df):
    # Winsorize (cap, don't drop) numeric feature columns using IQR bounds.
    # Excludes target and pure calendar/index-like fields.
    exclude = {"aqi_target", "day", "month", "day_of_week", "is_weekend",
               "hour", "station_latitude", "station_longitude"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    capped_counts = {}
    for c in numeric_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[c] < lower) | (df[c] > upper)).sum())
        if n_out:
            capped_counts[c] = n_out
        df[c] = df[c].clip(lower=lower, upper=upper)

    log.append(f"Outliers capped (IQR method) in columns: {capped_counts if capped_counts else 'none'}.")
    return df


def drop_uninformative_columns(df):
    dropped = []

    # Always keep these even if constant in a small sample: required
    # time-based features, derived AQI features, city, and timestamp.
    # Core sensor measurements are ALSO protected -- a column reading zero
    # variance in one small/skewed batch (e.g. UVI is genuinely 0 at night)
    # does not mean the feature is uninformative overall, and dropping it
    # breaks the fixed downstream Hopsworks schema, which expects every
    # measurement column to be present on every insert.
    protected = {
        "date", "aqi_target", "city", "timestamp",
        "hour", "day", "month", "day_of_week", "is_weekend",
        "aqi_lag_1", "aqi_change_rate", "aqi_rolling_mean_3",
        "o3_avg", "o3_max",
        "pm10_avg", "pm10_max", "pm10_min",
        "pm25_avg", "pm25_max", "pm25_min",
        "uvi_avg", "uvi_max",
    }

    # Zero-variance columns add nothing for training (excluding protected ones)
    for c in df.columns:
        if c in protected:
            continue
        if df[c].nunique(dropna=False) <= 1:
            dropped.append(c)
    df = df.drop(columns=[c for c in dropped if c in df.columns], errors="ignore")
    log.append(f"Dropped uninformative/zero-variance columns: {dropped if dropped else 'none'}.")
    log.append(f"Kept time-based features: hour, day, month, day_of_week, is_weekend.")
    log.append(f"Kept derived AQI features: aqi_lag_1, aqi_change_rate, aqi_rolling_mean_3.")
    log.append(f"Kept city and timestamp columns.")
    return df


def encode_categoricals(df):
    # NOTE: previously this one-hot encoded 'dominant_pollutant' with
    # pd.get_dummies(), but that creates a DIFFERENT set of columns
    # depending on which pollutant categories happen to appear in THIS
    # batch. Since Hopsworks locks the feature group schema on first
    # insert, an hourly batch that's missing (or introduces a new)
    # category would break every future upload the same way the
    # zero-variance bug did. One-hot encoding belongs at model-training
    # time (where you control the full category list), not in the shared
    # feature-store ingestion path, where the schema must stay fixed.
    # 'dominant_pollutant' is kept as-is (already lowercase/stripped in
    # fix_dtypes) so it lands in Hopsworks as a plain string feature.
    log.append("Left categorical columns (e.g. dominant_pollutant) as plain "
                "strings -- one-hot encoding is deferred to training time to "
                "keep the Hopsworks schema stable across batches.")
    return df


def round_numeric(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].round(2)
    log.append(f"Rounded numeric columns to 2 decimal places: {numeric_cols}.")
    return df


def finalize(df):
    # Ensure target has no missing rows (can't train without label)
    if "aqi_target" in df.columns:
        before = df.shape[0]
        df = df.dropna(subset=["aqi_target"])
        after = df.shape[0]
        if before != after:
            log.append(f"Dropped {before - after} row(s) missing the target variable.")
    else:
        log.append("Warning: 'aqi_target' column not found — skipping target-completeness check.")

    df = df.sort_values("date").reset_index(drop=True)

    # Replace 'date' with 'timestamp' in the same leading position
    if "date" in df.columns and "timestamp" in df.columns:
        cols = df.columns.tolist()
        cols.remove("date")
        cols.remove("timestamp")
        df = df[["timestamp"] + cols]
        log.append("Removed 'date' column; 'timestamp' now leads the column order.")
    elif "date" in df.columns:
        cols = df.columns.tolist()
        cols.remove("date")
        df = df[["date"] + cols]
        log.append("No 'timestamp' column available; kept 'date' as the leading column.")

    return df


def main():
    df = load_data(INPUT_FILE)
    df = fix_dtypes(df)
    df = drop_duplicates(df)
    df = handle_missing(df)
    df = cap_outliers(df)
    df = drop_uninformative_columns(df)
    df = encode_categoricals(df)
    df = round_numeric(df)
    df = finalize(df)

    # mode="w" -> file is fully overwritten, no stale rows left behind
    with open(OUTPUT_FILE, "w", newline="") as f:
        df.to_csv(f, index=False)

    log.append(f"Saved cleaned dataset -> {OUTPUT_FILE} ({df.shape[0]} rows, {df.shape[1]} columns).")

    print("\n".join(log))


if __name__ == "__main__":
    main()