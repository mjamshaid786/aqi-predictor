import os
import hopsworks
from dotenv import load_dotenv

# 1. .env file se environment variables load karein
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor_2026")

if not HOPSWORKS_API_KEY:
    print("❌ Error: HOPSWORKS_API_KEY .env file mein nahi mili!")
    exit(1)

print("⚡ Hopsworks Feature Store se connect ho rahe hain...")

# 2. Hopsworks login
try:
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT
    )
    fs = project.get_feature_store()
    print("✅ Hopsworks connection successful!")
except Exception as e:
    print(f"❌ Connection error: {e}")
    exit(1)

# 3. Feature Group ka naam aur version (Apne feature group ke mutabiq name adjust kar lein agar different ho)
FEATURE_GROUP_NAME = "aqi_predictions"  # Agar aap ka name alag hai toh yahan change kar lein
FEATURE_GROUP_VERSION = 6

try:
    print(f"📊 Feature group '{FEATURE_GROUP_NAME}' se data fetch ho raha hai...")
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
    # Hopsworks se data Pandas DataFrame mein read karna
    df = fg.read()

    # 4. Total rows aur Last 5 rows print karna
    print("\n" + "=" * 50)
    print(f"📈 TOTAL ROWS IN HOPSWORKS: {len(df)}")
    print("=" * 50)

    print("\n📌 LAST 5 ROWS:")
    print(df.tail(5))
    print("=" * 50)

except Exception as e:
    print(f"❌ Data read karne mein error aya: {e}")