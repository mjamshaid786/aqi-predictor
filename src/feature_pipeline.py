import os
import sys
import time
import logging
import logging.handlers
import traceback
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / "aqi_pipeline.log"

OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CITIES: List[str] = ["lahore"]
AQICN_BASE_URL = "https://api.waqi.info/feed/{city}/"
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("aqi_pipeline")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(LOG_FILE_PATH), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class StepTracker:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.results: List[Dict[str, Any]] = []

    @contextmanager
    def step(self, name: str, critical: bool = False):
        start_time = time.perf_counter()
        self.logger.info(f"--> STARTING step: '{name}'")
        try:
            yield
            duration = time.perf_counter() - start_time
            self.logger.info(f"<-- SUCCESS step: '{name}' (took {duration:.2f}s)")
            self.results.append({"step": name, "status": "PASS", "duration_sec": round(duration, 2), "error": None})
        except Exception as exc:
            duration = time.perf_counter() - start_time
            self.logger.error(f"<-- FAILED step: '{name}' (after {duration:.2f}s): {exc}")
            self.logger.debug(traceback.format_exc())
            self.results.append({"step": name, "status": "FAIL", "duration_sec": round(duration, 2), "error": str(exc)})
            if critical:
                raise

    def print_summary(self) -> None:
        self.logger.info("=" * 78)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 78)
        for result in self.results:
            line = f"[{result['status']}] {result['step']:<40} | {result['duration_sec']:>6.2f}s"
            if result["error"]:
                line += f" | ERROR: {result['error']}"
            self.logger.info(line)
        self.logger.info("=" * 78)


def load_environment(logger: logging.Logger) -> Dict[str, Optional[str]]:
    load_dotenv(find_dotenv())

    aqicn_key = os.getenv("AQICN_API_KEY")
    if not aqicn_key:
        raise RuntimeError("AQICN_API_KEY is missing from .env")

    logger.info("Environment variables loaded.")
    return {"aqicn_key": aqicn_key}


class AQICNClient:
    def __init__(self, api_key: str, logger: logging.Logger):
        self.api_key = api_key
        self.logger = logger

    def fetch_city_feed(self, city: str) -> Dict[str, Any]:
        url = AQICN_BASE_URL.format(city=city)
        params = {"token": self.api_key}
        last_exception: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                payload = response.json()
                if payload.get("status") != "ok":
                    raise RuntimeError(f"AQICN status='{payload.get('status')}' for '{city}'")
                self.logger.info(f"[{city}] Fetched feed (AQI={payload['data'].get('aqi')}).")
                return payload["data"]
            except Exception as exc:
                last_exception = exc
                self.logger.warning(f"[{city}] Attempt {attempt} failed: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise RuntimeError(f"[{city}] All attempts failed: {last_exception}")

    def fetch_many(self, cities: List[str]) -> Dict[str, Dict[str, Any]]:
        feeds: Dict[str, Dict[str, Any]] = {}
        for city in cities:
            try:
                feeds[city] = self.fetch_city_feed(city)
            except Exception as exc:
                self.logger.error(f"Skipping city '{city}': {exc}")
        self.logger.info(f"Fetched {len(feeds)}/{len(cities)} cities.")
        return feeds


class FeatureEngineer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def build_city_dataframe(self, city: str, feed: Dict[str, Any]) -> pd.DataFrame:
        forecast_block = feed.get("forecast", {}).get("daily", {})
        if not forecast_block:
            self.logger.warning(f"[{city}] No forecast data available.")
            return pd.DataFrame()

        pollutant_frames = []
        for pollutant_name, daily_readings in forecast_block.items():
            if not isinstance(daily_readings, list):
                continue
            df_pollutant = pd.DataFrame(daily_readings)
            df_pollutant = df_pollutant.rename(columns={
                "avg": f"{pollutant_name}_avg", "max": f"{pollutant_name}_max", "min": f"{pollutant_name}_min",
            })
            df_pollutant["date"] = pd.to_datetime(df_pollutant["day"])
            df_pollutant = df_pollutant.drop(columns=["day"])
            pollutant_frames.append(df_pollutant.set_index("date"))

        if not pollutant_frames:
            return pd.DataFrame()

        merged = pd.concat(pollutant_frames, axis=1, join="outer").reset_index()
        merged["city"] = city
        merged["station_latitude"] = feed.get("city", {}).get("geo", [np.nan, np.nan])[0]
        merged["station_longitude"] = feed.get("city", {}).get("geo", [np.nan, np.nan])[1]
        merged["dominant_pollutant"] = feed.get("dominentpol", np.nan)
        merged["current_overall_aqi"] = feed.get("aqi", np.nan)

        self.logger.info(f"[{city}] Built {len(merged)} rows.")
        return merged

    def add_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df["date"].dt.hour
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        return df

    def add_derived_features(self, df: pd.DataFrame, value_col: str = "pm25_avg") -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["city", "date"]).reset_index(drop=True)
        grouped = df.groupby("city", group_keys=False)

        df["aqi_lag_1"] = grouped[value_col].shift(1)
        df["aqi_change_rate"] = ((df[value_col] - df["aqi_lag_1"]) / df["aqi_lag_1"].replace(0, np.nan)) * 100.0
        df["aqi_rolling_mean_3"] = grouped[value_col].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        return df

    def add_target(self, df: pd.DataFrame, value_col: str = "pm25_avg") -> pd.DataFrame:
        df = df.copy()
        grouped = df.groupby("city", group_keys=False)
        df["aqi_target"] = grouped[value_col].shift(-1)
        return df

    def build_feature_table(self, feeds: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        per_city_frames = []
        for city, feed in feeds.items():
            city_df = self.build_city_dataframe(city, feed)
            if not city_df.empty:
                per_city_frames.append(city_df)

        if not per_city_frames:
            raise RuntimeError("No usable AQICN data returned for any city.")

        combined = pd.concat(per_city_frames, ignore_index=True)
        combined = self.add_time_based_features(combined)
        combined = self.add_derived_features(combined, value_col="pm25_avg")
        combined = self.add_target(combined, value_col="pm25_avg")
        combined["ingested_at"] = pd.Timestamp(datetime.now(timezone.utc))

        before_rows = len(combined)
        combined = combined.dropna(subset=["aqi_target"]).reset_index(drop=True)
        self.logger.info(f"Dropped {before_rows - len(combined)} rows with no target. Final shape: {combined.shape}")
        return combined


def main() -> int:
    logger = configure_logging()
    tracker = StepTracker(logger)
    logger.info(f"Pipeline started. Log file: {LOG_FILE_PATH}")

    env_vars: Dict[str, Optional[str]] = {}
    feeds: Dict[str, Dict[str, Any]] = {}
    feature_table = pd.DataFrame()

    with tracker.step("Load environment variables", critical=True):
        env_vars = load_environment(logger)

    with tracker.step("Fetch raw AQI data from AQICN"):
        client = AQICNClient(api_key=env_vars["aqicn_key"], logger=logger)
        feeds = client.fetch_many(DEFAULT_CITIES)

    with tracker.step("Engineer features and target"):
        engineer = FeatureEngineer(logger=logger)
        feature_table = engineer.build_feature_table(feeds)

    with tracker.step("Save features to CSV"):
        if feature_table.empty:
            raise RuntimeError("Feature table is empty.")
        csv_path = OUTPUT_DIR / "aqi_features_lahore.csv"
        write_header = not csv_path.exists()
        feature_table.to_csv(csv_path, mode="w", header=True, index=False)
        logger.info(f"Saved: {csv_path} ({len(feature_table)} rows appended).")

    tracker.print_summary()
    return 1 if any(r["status"] == "FAIL" for r in tracker.results) else 0


if __name__ == "__main__":
    sys.exit(main())