"""
Diagnostic: isolate why aqi_neural_network_model fails to load.
Run standalone: python diagnose_nn.py
"""
import os
import shutil
import hopsworks
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    project=os.getenv("HOPSWORKS_PROJECT", "aqi_predictor_2026"),
    host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
    port=443,
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
)
mr = project.get_model_registry()

# Force a clean re-download
local_cache = os.path.expanduser("/tmp/hopsworks/models/aqi_predictor_2026/aqi_neural_network_model")
if os.path.exists(local_cache):
    shutil.rmtree(local_cache)
    print(f"Removed stale local cache: {local_cache}")

models = mr.get_models("aqi_neural_network_model")
print(f"Found {len(models)} version(s) of aqi_neural_network_model")

model_meta = models[0]
print(f"Downloading version {model_meta.version}...")
model_dir = model_meta.download()
print(f"Downloaded to: {model_dir}")

print("\nFiles present:")
for f in os.listdir(model_dir):
    full = os.path.join(model_dir, f)
    print(f"  - {f} ({os.path.getsize(full)} bytes)")

h5_path = os.path.join(model_dir, "model.h5")
scaler_path = os.path.join(model_dir, "scaler.pkl")
feat_path = os.path.join(model_dir, "feature_names.txt")

print(f"\nmodel.h5 present: {os.path.exists(h5_path)}")
print(f"scaler.pkl present: {os.path.exists(scaler_path)}")
print(f"feature_names.txt present: {os.path.exists(feat_path)}")

if os.path.exists(h5_path):
    print("\nTrying to load model.h5 with TensorFlow/Keras...")
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        m = keras_load_model(h5_path, compile=False)  # skip broken legacy metrics deserialization
        print("✅ SUCCESS — model.h5 loaded fine.")
        m.summary()
    except Exception as e:
        print(f"❌ FAILED to load model.h5: {e}")
        import traceback
        traceback.print_exc()

if os.path.exists(scaler_path):
    print("\nTrying to load scaler.pkl...")
    try:
        import joblib
        s = joblib.load(scaler_path)
        print(f"✅ SUCCESS — scaler loaded: {s}")
    except Exception as e:
        print(f"❌ FAILED to load scaler.pkl: {e}")

        