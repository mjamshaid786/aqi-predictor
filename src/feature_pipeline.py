'''
==================================================
            IMPORTING REQUIRED LIBRARIES
==================================================
'''
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv

'''
-----------------------------------------------
            SETTING PROJECT BASIC PATHS
-----------------------------------------------             
'''
PROJECT_ROOT = Path(__file__).resolve().parent 
# EsSe Root Folder Select Ho JayeGa Ta K Baqi Paths Ko EsKy Relative Bna Sakein

OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 
#data Naam Ka Folder Jahan Hum Data Fetch Kr K Store KareinGy...

DEFAULT_CITIES: List[str] = ["lahore"] 
# Yahan Hum Un Cities K List Add Krtay Hain JinKa Data Hum Ne Fetch Krna Ho
AQICN_BASE_URL = "https://api.waqi.info/feed/{city}/" 
# Ye AQICN Se API Ki Help Se Humaray City Ka Data Nikalay Gi
REQUEST_TIMEOUT_SECONDS = 15 
#Agr 15 Seconde Tk Wait KareGa HTTP K Response Ka
MAX_RETRIES = 3 
#Agr API Call Fail Ho Jaye Tu 3 Baar Koshish Krni Hai
RETRY_BACKOFF_SECONDS = 2 
# Fail Honay K Baad Kitni Der Wait Kr K Dobara Try Krna Hai 
# (Yahan 2 Mtlb 2 Seconds Aur Oper K Code Mn Bhi Same Seconds Format Hoga)
'''
==================================================
        SETTING API KEY FOR FETCHING DATA
==================================================
'''
def load_environment() -> Dict[str, Optional[str]]:
#Ye Function .env Se Environmental  Vairaables Load Krta Hai Aur UnKo Dict. Form Return Krta Hai
    load_dotenv(find_dotenv())
    #find_donenv() project mn .env file search krta hai
    #load_dotenv() ye usay python k environmental variaables mn load krta hai
    #Ta K os.getenv() UsKo Use Kr Skay
 
    aqicn_key = os.getenv("AQICN_API_KEY")
    #Ye Us Environment Se "AQICN_API_KEY" Variable Ka Samnay Ji API Key Hogi Read KryGa
    if not aqicn_key:
        raise RuntimeError("AQICN_API_KEY is missing from .env")
    #Agr API Key Nahi Milti Tu Error Show Hoga
 
    return {"aqicn_key": aqicn_key}
    #Agr Mil Gai Tu Dict. Ki Form Mn Return Ho JayeGi
'''
==============================
        FETCHING DATA
==============================
'''
 
class AQICNClient:
# Ye Class API Se Data Fetch Krnay K Liye Hai...
    def __init__(self, api_key: str):
        self.api_key = api_key
        # API Key Ko Instance Variable Mn Store Kr Rahe Hain..
 
    def fetch_city_feed(self, city: str) -> Dict[str, Any]:
    #Ye Method Humari City Ka Data Fetch KareGa.
        url = AQICN_BASE_URL.format(city=city)
        #url Mn Actual City Ka URL Store Ho JayeGa.
        params = {"token": self.api_key}
        #Query Parameters Mn Tokken Pass KareinGy Authentication K Liye.
        last_exception: Optional[Exception] = None
        # Jb Code Re-Try Krnay K Baad Bhi Data Fetch Nahi Kr Skay Ga Tu Last Pe
        # Jo Error HoGa Wo Yahan Store Hoga.
 
        for attempt in range(1, MAX_RETRIES + 1):
        # Ye Loop 3 Baar Try KareGa Jaisay Mene set Kiya Tha Shoro Mn...
            try:
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
                # Data K Liye Request Send Ki Hai Timeout K Sath.
                response.raise_for_status()
                # Agr response Mn Koi Error Hoga Tu Code Seedha Exception Pr Chala JayeGa.
                payload = response.json()
                # Ye Reponse Walay Data Ko Dekhay Ga Andar Kia Hai Aur Simple
                # dict Mn Change KareGa.
                if payload.get("status") != "ok":
                    raise RuntimeError(f"AQICN status='{payload.get('status')}' for '{city}'")
                return payload["data"]
            except Exception as exc:
                last_exception = exc
                if attempt < MAX_RETRIES:
                    import time
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
 
        raise RuntimeError(f"[{city}] All attempts failed: {last_exception}")
 
    def fetch_many(self, cities: List[str]) -> Dict[str, Dict[str, Any]]:
    #Ye Function Multiple Cities Ka Data Fetch KareGa Aur Har City K Data K Liye
    # fetch_city_feed Wala Function Call Hoga.
        feeds: Dict[str, Dict[str, Any]] = {}
        for city in cities:
            try:
                feeds[city] = self.fetch_city_feed(city)
            except Exception as exc:
                print(f"Skipping city '{city}': {exc}")
        return feeds
 
'''
 =====================================================
        CLEANING DATA FOR DESIRED COLUMNS
======================================================
'''
class FeatureEngineer:
#Ye Class AQICN Se Fetch Kiye Data Ko Desired Format Mn Change KareyGi (Columns)
    def build_city_dataframe(self, city: str, feed: Dict[str, Any]) -> pd.DataFrame:
    # Ye Method Data Ko Dict. Se Pandas K DataFrame Mn Convert KareGi
        forecast_block = feed.get("forecast", {}).get("daily", {})
        # Feed Se Daily Forcast Wala Hissa NikalyGa Jaisa K pm25, pm10 etc
        # Agr Kuch Nahi Milta Tu Khali Dict MileGi.
        if not forecast_block:
            return pd.DataFrame()
            #Agr Forecast Wala Data Khali Hai Tu Khali DataFrame Return KarenGy.
 
        pollutant_frames = []
        # Har Pollutant Ka Chhota dataFrame Yahan Jama Hoga
        for pollutant_name, daily_readings in forecast_block.items():
            if not isinstance(daily_readings, list):
                continue
            # Agr Daily Reading Dict. Format Mn Nahi Hai Tu Skip Ho JayeGa.
            df_pollutant = pd.DataFrame(daily_readings)
            df_pollutant = df_pollutant.rename(columns={
                "avg": f"{pollutant_name}_avg", "max": f"{pollutant_name}_max", "min": f"{pollutant_name}_min",
            })
            # Column Ko Rename Kr Rahe Hain Ta K Jb Sb Pollutants Merge Hoon Tu
            # Column Class Na Ho Jaye.
            df_pollutant["date"] = pd.to_datetime(df_pollutant["day"])
            # day column(str) Ko Proper datetime format Mn Convert Kr K date Mn Store Hoga.
            df_pollutant = df_pollutant.drop(columns=["day"])
            pollutant_frames.append(df_pollutant.set_index("date"))
            #date Ko index Bna Kr EsKo List Mn Add KareGa Ta K 
            #Fr Baad Mn date K Hissab Se Merge Kr Skein.
 
        if not pollutant_frames:
            return pd.DataFrame()
            # Agr Koi Valid Pollutant Frame Nahi Bna Tu Khali DataFrame Return Ho JayeGa.
 
        merged = pd.concat(pollutant_frames, axis=1, join="outer").reset_index()
        # Sab pollutants Ke DataFrames Ko columns Ki Tarah (axis=1) side by
        # side Jor Rahe Hain, "outer" join Se Koi Bhi date Miss Nahi Hoga.
        # reset_index() Se "date" Wapis normal column Bann Jata Hai.
        merged["city"] = city
        # Naya Column city Add Kiya Hai Tak Har Row Mn City Ka Naam Save Ho Skay.
        merged["station_latitude"] = feed.get("city", {}).get("geo", [np.nan, np.nan])[0]
        # Feed K AndarSe Monitoring Station Ki latitude Nikal Kar Naye
        # Column Mein Daal Rahe Hain (agar geo data na ho to NaN).
        merged["station_longitude"] = feed.get("city", {}).get("geo", [np.nan, np.nan])[1]
        merged["dominant_pollutant"] = feed.get("dominentpol", np.nan)
        # Feed se "dominentpol" (Sab Se Ziyada Asar Dalne Wala Pollutant)
        # Nikal Kar Column Mein Daal Rahe Hain.
        merged["current_overall_aqi"] = feed.get("aqi", np.nan)
        # Feed Se Current Overall AQI Value Nikal Kar Column Mein Daal Dega.
        return merged
 
    def add_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        #Ye Method Humein Jo Hour, Day, Month Wagera Chahiye Wo Niklay Ga.
        df = df.copy()
        df["hour"] = df["date"].dt.hour
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        # Agar day_of_week 5 (Saturday) ya 6 (Sunday) hai to True, warna
        # False — phir isay 1/0 integer mein convert kar rahe hain, taake
        # yeh weekend indicator feature ban sake.
        return df
 
    def add_derived_features(self, df: pd.DataFrame, value_col: str = "pm25_avg") -> pd.DataFrame:
        # Yeh method lag, percentage change, aur rolling average jaisi
        # derived (nikali hui) features banata hai.
        df = df.copy()
        df = df.sort_values(["city", "date"]).reset_index(drop=True)
        # Data ko pehlay city ke hisab se, phir date ke hisab se sort kar
        # rahe hain — is se lag/rolling features sahi order mein calculate
        # hongi. reset_index taake naya clean index mil jaye.
        grouped = df.groupby("city", group_keys=False)
        # Data ko city ke hisab se group kar rahe hain, taake har city ka
        # data alag se process ho (ek city ka lag doosri city ke data se
        # mix na ho). group_keys=False result ko flat rakhta hai.
 
        df["aqi_lag_1"] = grouped[value_col].shift(1)
        # Har city ke andar value_col (jaise pm25_avg) ki 1 din pehlay wali
        # value nikal kar naya column "aqi_lag_1" bana rahe hain (lag
        # feature).
        df["aqi_change_rate"] = ((df[value_col] - df["aqi_lag_1"]) / df["aqi_lag_1"].replace(0, np.nan)) * 100.0
        # Aaj ki value aur pichlay din ki value ke darmiyan percentage
        # change calculate kar rahe hain. Agar lag value 0 ho to usay NaN
        # se replace kar rahe hain taake division-by-zero error na aaye.
        df["aqi_rolling_mean_3"] = grouped[value_col].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        # Har City Ke Andar 3-din Ka moving/rolling average calculate Kar
        # Rahe Hain (min_periods=1 matlab shuru ke 1-2 din bhi average mil
        # jayega, jitna data available ho usi se).
        return df
    '''
    ===============================================================
            SETTING REQUIREN COLUMNS FOR MACHINE LEARNING
    ===============================================================
    ''' 
    def add_target(self, df: pd.DataFrame, value_col: str = "pm25_avg") -> pd.DataFrame:
    # Yeh Method Machine Learning Ke Liye "target" Column Banata Hai
    # (Jo Predict Karna Hai).
        df = df.copy()
        grouped = df.groupby("city", group_keys=False)
        df["aqi_target"] = grouped[value_col].shift(-1)
        # Har City Ke Andar "Agle Din" Ki Value_col Value Nikal Kar
        # "aqi_target" column Bana Rahe Hain — Ye Future Value Hai Jo
        # Model Ne Predict Karni Hai (shift(-1) Matlab Ek row Upar Se Value
        # Le Rahe Hain).
        return df
 
    def build_feature_table(self, feeds: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    # Yeh method Sab Cities Ke Feeds Ko Le Kar Ek Final, ML-ready
    # Feature table Banata Hai.
        per_city_frames = []
        for city, feed in feeds.items():
            city_df = self.build_city_dataframe(city, feed)
            # Us city Ka raw feed clean DataFrame Mein Convert Kar Rahe Hain.
            if not city_df.empty:
                per_city_frames.append(city_df)
 
        if not per_city_frames:
            raise RuntimeError("No usable AQICN data returned for any city.")
 
        combined = pd.concat(per_city_frames, ignore_index=True)
        # Sab cities Ke DataFrames Ko Rows Ki Tarah (Ek Ke Neeche Ek) Jor
        # Kar Ek Combined DataFrame Bana Rahe Hain. ignore_index=True Se
        # Naya Clean Index Milega.
        combined = self.add_time_based_features(combined)
        combined = self.add_derived_features(combined, value_col="pm25_avg")
        combined = self.add_target(combined, value_col="pm25_avg")
        combined["ingested_at"] = pd.Timestamp(datetime.now(timezone.utc))
        # Ek Column Add Kar Rahe Hain Jismein Ye Record Hoga Ke Ye Data
        # Kis Waqt (UTC Mein) fetch/process Kiya Gaya.
 
        combined = combined.dropna(subset=["aqi_target"]).reset_index(drop=True)
        # Un Rows Ko Hata Rahe Hain JinKa "aqi_target" Khali (NaN) Hai —
        # Kyunki Agla Din Ka Data Na Hone Ki Wajah Se Inn Rows Ko Training
        # Ke Liye Use Nahi Kiya Ja Sakta. reset_index Se Index Dobara Clean
        # Ho Jata Hai.
        return combined
 
 
def main() -> int:
    print("Pipeline started.")
 
    print("Step: Loading environment variables...")
    env_vars = load_environment()
    # Environment Variables Load Karne Wala Step — critical=True Matlab
    # Agar Ye fail Ho To Poora Program Yahin Ruk Jayega (Aage Jane Ka Koi
    # Faida Nahi Kyunki API Key Ke Baghair Kuch Nahi Ho Sakta).
 
    print("Step: Fetching raw AQI data from AQICN...")
    client = AQICNClient(api_key=env_vars["aqicn_key"])
    feeds = client.fetch_many(DEFAULT_CITIES)
 
    print("Step: Engineering features and target...")
    engineer = FeatureEngineer()
    feature_table = engineer.build_feature_table(feeds)
 
    print("Step: Saving features to CSV...")
    if feature_table.empty:
        raise RuntimeError("Feature table is empty.")
    csv_path = OUTPUT_DIR / "aqi_features_lahore.csv"
    feature_table.to_csv(csv_path, mode="w", header=True, index=False)
    print(f"Saved: {csv_path} ({len(feature_table)} rows).")
 
    print("Pipeline finished successfully.")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
    # Yeh Check Karta Hai Ke Script Ko Directly Run Kiya Gaya Hai (Na Ke
    # Kisi Doosre Module Se Import Kiya Gaya Hai). Agar Directly Run Hua Hai
    # Mo main() Function Chalao Aur Uska Return Value (Exit Code) Ko
    # sys.exit() Ke Zariye Operating System Ko Wapis Bhej Do.
