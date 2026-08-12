# 🌫️ Lahore AQI Predictor — End-to-End MLOps Pipeline

[![Python Version](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hopsworks Feature Store](https://img.shields.io/badge/Hopsworks-Feature%20Store-00A88F?style=for-the-badge)](https://www.hopsworks.ai/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Automated%20Pipelines-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io/)

A production-grade **Air Quality Index (AQI) Prediction System** built with modern **MLOps practices**. The system ingests hourly pollutant and weather data from the **Open-Meteo API**, engineers features, stores them in a **Hopsworks Feature Store**, retrains multi-horizon prediction models via CI/CD pipelines, and serves real-time and 3-day-ahead predictions through an interactive **Streamlit** dashboard — with SHAP-based model explainability built in.

---

## 🏗️ System Architecture & MLOps Workflow

The project follows a modular, decoupled MLOps architecture split into four pipelines:

```
              ┌──────────────────────────┐
              │   Open-Meteo API         │
              │ (Air Quality + Weather)  │
              └────────────┬─────────────┘
                            │ Hourly raw data
                            ▼
┌───────────────────────────────────────────────────────────┐
│                 Hourly Feature Pipeline                    │
│      (GitHub Actions: src/feature_pipeline.py)             │
│  Fetches pollutants + weather → engineers features →       │
│  upserts directly into the Hopsworks Feature Store         │
└───────────────────────────┬─────────────────────────────────┘
                             ▼
                  ┌──────────────────────┐
                  │  Hopsworks Feature   │
                  │        Store         │
                  └───────────┬──────────┘
                              │ (features, multi-horizon targets)
                              ▼
┌───────────────────────────────────────────────────────────┐
│               Daily Model Training Pipeline                 │
│        (GitHub Actions: src/training_pipeline.py)          │
│  Trains 5 algorithms × 4 forecast horizons                 │
│  (next-hour, +24h, +48h, +72h) and registers each           │
└───────────────────────────┬─────────────────────────────────┘
                             ▼
                  ┌──────────────────────┐
                  │   Hopsworks Model    │
                  │       Registry       │
                  └───────────┬──────────┘
                              │ (per-horizon best models)
                              ▼
┌───────────────────────────────────────────────────────────┐
│                   Inference + Web App                       │
│   src/inference_pipeline.py + app.py (Streamlit)            │
│  Loads the best model per horizon, predicts from the        │
│  latest real feature row, converts PM2.5 → true AQI,        │
│  and renders live status + real 3-day forecast + SHAP        │
└───────────────────────────────────────────────────────────┘
```

1. **Feature Pipeline (Hourly):**
   - GitHub Action fetches the latest hourly pollutant (`PM2.5`, `PM10`, `O3`) and weather (temperature, humidity, wind speed/direction, pressure, precipitation, cloud cover) data from **Open-Meteo** — the *same source and hourly resolution* used for backfill, so there is no train/serve data mismatch.
   - Engineers time-based features (hour/day/month/day-of-week), lag/rolling-mean features, and upserts directly into the Hopsworks Feature Store.
2. **Backfill (`src/backfill_data.py`):**
   - Populates historical training data (up to ~2 years, automatically falling back to a smaller window if Open-Meteo's archive doesn't go back that far) with the same feature engineering as the live pipeline.
3. **Training Pipeline (Daily):**
   - Fetches historical (features, targets) from the Feature Store.
   - Trains **5 algorithms** (Random Forest, Gradient Boosting, Ridge, Lasso, Neural Network) for **4 real forecast horizons**: next-hour, +24h, +48h, and +72h — each trained against its own genuine future ground-truth target (not a single next-step model reused recursively).
   - Evaluates every model with RMSE, MAE, R², and MAPE, and registers all of them to the Hopsworks Model Registry.
4. **Inference + Dashboard:**
   - `src/inference_pipeline.py` loads the best-performing model *for each horizon* (lowest test RMSE) and predicts directly from the latest real feature row — no simulated or randomly-perturbed values.
   - Raw PM2.5 model output is converted to the real **US EPA AQI index** via the standard breakpoint formula, so displayed numbers match public AQI sources.
   - The Streamlit app (`app.py`) shows live AQI status, a real 3-day forecast, a historical trend chart, and **SHAP** explainability plots (model-agnostic, works for any of the 5 algorithms).

---

## ✨ Key Features

* **🤖 Fully Automated MLOps:** Hourly feature refresh + daily retraining via GitHub Actions, no manual steps required.
* **📅 True Multi-Horizon Forecasting:** Separate models trained on real +24h/+48h/+72h ground truth, not a next-step model chained recursively.
* **📦 Centralized Feature Management:** Hopsworks Feature Store keeps training and serving features identical (same Open-Meteo source, same schema).
* **🔍 Explainable AI:** SHAP-based feature importance and beeswarm plots, computed consistently regardless of which algorithm is currently the best model.
* **📈 Dynamic Visualizations:** Interactive Plotly dashboards showing historical trend, live AQI, and forecast points, all on a consistent real-AQI scale.
* **🛡️ Resilient Pipelines:** Automatic retry logic for transient Hopsworks/Arrow Flight errors, and graceful fallback across models if a specific registry entry is unavailable.

---

## 🛠️ Tech Stack

* **Language:** Python 3.12
* **Data Manipulation & Analytics:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, TensorFlow/Keras
* **Explainability:** SHAP, Matplotlib
* **Data Visualization:** Plotly, Streamlit
* **Feature Store & Model Registry:** Hopsworks
* **CI/CD & Orchestration:** GitHub Actions
* **Data Provider:** Open-Meteo (Air Quality API + Weather API)

---

## 📂 Repository Structure

```text
aqi-predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml     # Hourly: fetch + upsert to Hopsworks
│       └── training_pipeline.yml    # Daily: multi-horizon training + registration
├── src/
│   ├── backfill_data.py             # Historical backfill (Open-Meteo, pollutants + weather)
│   ├── feature_pipeline.py          # Live hourly feature refresh (same source as backfill)
│   ├── uploading_to_hopsworks.py    # Feature Store schema creation/evolution + upload
│   ├── training_pipeline.py         # Multi-horizon model training & registry upload
│   └── inference_pipeline.py        # Real per-horizon inference logic (used by app.py)
├── app.py                           # Streamlit dashboard (live AQI, 3-day forecast, SHAP)
├── requirements.txt                 # Project dependencies
├── .gitignore
└── README.md                        # Documentation
```

---

## 🚀 Local Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mjamshaid786/aqi-predictor.git
cd aqi-predictor
```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```ini
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT=aqi_predictor_2026
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
```

> Note: no external pollution-API key is required — Open-Meteo's air quality and weather endpoints are used without authentication.

### 5. Run the Pipelines (first-time setup)

```bash
python src/backfill_data.py          # generates historical training data
python src/uploading_to_hopsworks.py # creates/updates the Feature Store schema and uploads it
python src/training_pipeline.py      # trains and registers all horizon models
```

### 6. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

---

## 🔐 Environment Secrets & Configuration

To enable the GitHub Actions pipelines, configure the following secrets under **Repository Settings → Secrets and variables → Actions**:

| Secret Name | Description |
| --- | --- |
| `HOPSWORKS_API_KEY` | User API token generated from Hopsworks |
| `HOPSWORKS_PROJECT` | Active Hopsworks project identifier (`aqi_predictor_2026`) |
| `HOPSWORKS_HOST` | Hopsworks cluster host (`eu-west.cloud.hopsworks.ai`) |

No pollution-data-provider API key is needed, since Open-Meteo's public endpoints don't require authentication.

---

## 🤝 Author & Acknowledgments

* **Developer:** Muhammad Jamshaid
* **GitHub:** [@mjamshaid786](https://github.com/mjamshaid786)
* **Special Thanks:** Built as part of an advanced hands-on MLOps implementation utilizing open-source infrastructure.