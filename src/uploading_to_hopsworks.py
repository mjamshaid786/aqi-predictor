
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    import hopsworks
    from hsfs.feature import Feature
except ImportError:
    print(
        "The 'hopsworks' package (and its 'hsfs'/'pyarrow' dependencies) is "
        "not fully installed.\n"
        "Install it with:  pip install hopsworks pyarrow"
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_PATH = Path("src/data/cleaned_data.csv")
FEATURE_GROUP_NAME = "aqi_predictions"
FEATURE_GROUP_VERSION = 6
FEATURE_GROUP_DESCRIPTION = "Hourly AQI prediction features per city (AQICN source)"
SOURCE_TIMESTAMP_COLUMN = "timestamp"  # raw per-row observation timestamp from AQICN
PRIMARY_KEY = ["city", "date", "hour"]  # one row per city, per calendar day, per hour
EVENT_TIME_COLUMN = "timestamp"         # full timestamp used for point-in-time correctness

# The AQICN feed is per-city. Set a fallback here (or via env var) only as a
# safety net -- the real fix is always to have the extraction step attach the
# actual city returned by the API.
DEFAULT_CITY = os.getenv("AQICN_CITY", "unknown_city")

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hopsworks-uploader")


def load_environment() -> str:
    """Load .env file and return the Hopsworks API key."""
    logger.info("Loading environment variables from .env ...")
    load_dotenv()  # looks for .env in the current working directory by default

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        logger.error(
            "HOPSWORKS_API_KEY not found. Make sure your .env file contains:\n"
            "    HOPSWORKS_API_KEY=your_key_here"
        )
        sys.exit(1)

    logger.info("API key loaded successfully. [1/5]")
    return api_key


def sanitize_integer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce explicit, consistent dtypes so the Hopsworks schema never has to
    guess a type from whatever values happen to be in a given hourly batch.

    Pollutant/UV measurements are continuous readings that can legitimately
    include decimals (an "average" of several sensor readings is rarely a
    whole number) -- these are always cast to float64/double, even if a
    particular batch happens to contain whole numbers. Calendar/time-derived
    fields are always whole numbers by construction, so those are cast to a
    genuine int64 (bigint) rather than left to numpy's platform-dependent
    int32 default.
    """
    float_columns = [
        "o3_avg", "o3_max",
        "pm10_avg", "pm10_max", "pm10_min",
        "pm25_avg", "pm25_max", "pm25_min",
        "uvi_avg", "uvi_max",
        "aqi_lag_1", "aqi_change_rate", "aqi_rolling_mean_3", "aqi_target",
    ]
    int_columns = ["hour", "day", "month", "day_of_week", "is_weekend"]

    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype("int64")

    return df


def align_to_feature_group_schema(df: pd.DataFrame, feature_group) -> pd.DataFrame:
    """
    Add any column the feature group's schema expects but which is absent
    from this batch (e.g. the AQICN response didn't include UVI data this
    hour), filled with nulls, so the insert doesn't fail with a "missing
    from input dataframe" error. Existing columns are left untouched.
    """
    existing = {c.lower() for c in df.columns}
    for feature in feature_group.features:
        if feature.name.lower() not in existing:
            logger.warning(
                "Column '%s' expected by the feature group schema but "
                "missing from this batch. Adding it as null.",
                feature.name,
            )
            df[feature.name] = pd.NA
    return df


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Read and lightly validate the CSV before upload."""
    logger.info("Reading dataset from '%s' ...", csv_path)

    if not csv_path.exists():
        logger.error("File not found: %s", csv_path.resolve())
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if df.empty:
        logger.error("The CSV file is empty. Nothing to upload.")
        sys.exit(1)

    # Parse the raw observation timestamp -- this is the source of truth for
    # both the event-time column and the derived 'date' / 'hour' key fields.
    if SOURCE_TIMESTAMP_COLUMN not in df.columns:
        logger.error(
            "Required column '%s' not found in CSV. Found columns: %s",
            SOURCE_TIMESTAMP_COLUMN,
            list(df.columns),
        )
        sys.exit(1)

    df[SOURCE_TIMESTAMP_COLUMN] = pd.to_datetime(df[SOURCE_TIMESTAMP_COLUMN])

    # Always derive 'date' and 'hour' FROM the timestamp, overwriting any
    # pre-existing values for those columns. This guarantees the primary key
    # stays correct even if an upstream step wrote a stale/duplicate 'hour'.
    df["date"] = df[SOURCE_TIMESTAMP_COLUMN].dt.date.astype(str)
    df["hour"] = df[SOURCE_TIMESTAMP_COLUMN].dt.hour

    # 'city' is required for the composite primary key ['city', 'date', 'hour'].
    # If the upstream extraction step hasn't attached it yet, fill it in here
    # so uploads don't fail -- but this should ideally come from the AQICN
    # response itself (city.name) rather than a hardcoded default.
    if "city" not in df.columns:
        logger.warning(
            "'city' column not found in CSV. Filling with default city '%s'. "
            "Update your extraction pipeline to include the real city name.",
            DEFAULT_CITY,
        )
        df.insert(0, "city", DEFAULT_CITY)

    # Guard against duplicate primary keys, which Hopsworks will upsert
    # (overwrite) rather than append -- exactly what hourly ingestion must avoid.
    dupes = df.duplicated(subset=PRIMARY_KEY).sum()
    if dupes:
        logger.warning(
            "%d duplicate rows found for primary key %s. "
            "Hopsworks will upsert these, keeping the latest value.",
            dupes,
            PRIMARY_KEY,
        )

    df = sanitize_integer_columns(df)

    logger.info(
        "Dataset loaded: %d rows x %d columns. [2/5]", df.shape[0], df.shape[1]
    )
    return df


def connect_to_hopsworks(api_key: str):
    """Authenticate and return the Hopsworks feature store handle."""
    logger.info("Connecting to Hopsworks ...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    logger.info("Connected to project '%s'. [3/5]", project.name)
    return fs


def get_or_create_feature_group(fs, sample_df: pd.DataFrame):
    """
    Fetch an existing feature group, or create a new one with an EXPLICITLY
    declared schema.

    Passing `features=[...]` here (rather than letting Hopsworks infer types
    from whatever dataframe happens to be inserted first) is what actually
    fixes the recurring 'wrong type' errors: type inference depends on the
    values present in that specific batch (e.g. all-whole-number pollutant
    readings look like 'bigint' even though the column is conceptually a
    'double'). An explicit schema is fixed at creation time and never
    depends on which batch runs first.
    """
    logger.info(
        "Preparing feature group '%s' (v%d) ...",
        FEATURE_GROUP_NAME,
        FEATURE_GROUP_VERSION,
    )

    schema = [
        Feature(name="city", type="string"),
        Feature(name="date", type="string"),
        Feature(name="timestamp", type="timestamp"),
        Feature(name="hour", type="bigint"),
        Feature(name="day", type="bigint"),
        Feature(name="month", type="bigint"),
        Feature(name="day_of_week", type="bigint"),
        Feature(name="is_weekend", type="bigint"),
        Feature(name="o3_avg", type="double"),
        Feature(name="o3_max", type="double"),
        Feature(name="pm10_avg", type="double"),
        Feature(name="pm10_max", type="double"),
        Feature(name="pm10_min", type="double"),
        Feature(name="pm25_avg", type="double"),
        Feature(name="pm25_max", type="double"),
        Feature(name="pm25_min", type="double"),
        Feature(name="uvi_avg", type="double"),
        Feature(name="uvi_max", type="double"),
        Feature(name="aqi_lag_1", type="double"),
        Feature(name="aqi_change_rate", type="double"),
        Feature(name="aqi_rolling_mean_3", type="double"),
        Feature(name="aqi_target", type="double"),
    ]

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=FEATURE_GROUP_DESCRIPTION,
        primary_key=PRIMARY_KEY,
        event_time=EVENT_TIME_COLUMN,
        online_enabled=False,
        time_travel_format="HUDI",  # avoids requiring the local 'delta' package
        features=schema,
    )

    logger.info("Feature group ready. [4/5]")
    return feature_group


def upload_dataframe(feature_group, df: pd.DataFrame) -> None:
    """Insert the dataframe into the feature group and report progress."""
    total_rows = len(df)
    logger.info("Uploading %d rows to Hopsworks ...", total_rows)

    start = time.time()
    job, _ = feature_group.insert(df, write_options={"wait_for_job": True})
    elapsed = time.time() - start

    logger.info(
        "Upload complete: %d rows inserted in %.2f seconds. [5/5]",
        total_rows,
        elapsed,
    )

    if job is not None:
        logger.info("Ingestion job status: %s", job.get_state())


def main() -> None:
    print("=" * 60)
    print(" Hopsworks Feature Store Upload ")
    print("=" * 60)

    api_key = load_environment()
    df = load_dataset(DATA_PATH)
    fs = connect_to_hopsworks(api_key)
    feature_group = get_or_create_feature_group(fs, df)
    df = align_to_feature_group_schema(df, feature_group)
    upload_dataframe(feature_group, df)

    print("-" * 60)
    print(f"SUCCESS: '{DATA_PATH}' uploaded to Hopsworks "
          f"feature group '{FEATURE_GROUP_NAME}' (v{FEATURE_GROUP_VERSION}).")
    print("-" * 60)


if __name__ == "__main__":
    main()