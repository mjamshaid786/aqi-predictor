
# 🌫️ Lahore AQI Predictor — End-to-End MLOps Pipeline

[![Python Version](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hopsworks Feature Store](https://img.shields.io/badge/Hopsworks-Feature%20Store-00A88F?style=for-the-badge)](https://www.hopsworks.ai/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Automated%20Pipelines-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io/)

An production-grade **Air Quality Index (AQI) Prediction System** built with modern **MLOps practices**. This system automatically ingests live air quality data from the AQICN API, processes features, stores them in a **Hopsworks Feature Store**, regularly retrains the prediction model via CI/CD pipelines, and serves real-time predictions through an interactive **Streamlit** Web Application.

---

## 🏗️ System Architecture & MLOps Workflow

The project follows a modular and decoupled MLOps architecture split into three distinct pipelines:


```

```
              ┌──────────────────────┐
              │      AQICN API       │
              └──────────┬───────────┘
                         │ (Hourly Raw Data)
                         ▼

```

┌─────────────────────────────────────────────────────────┐
│               Hourly Feature Pipeline                   │
│   (GitHub Actions: src/feature_pipeline.py & clean.py)  │
└───────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────┐
│   Hopsworks Feature  │
│        Store         │
└───────────┬──────────┘
│ (Feature Groups)
▼
┌─────────────────────────────────────────────────────────┐
│               Daily Model Training Pipeline             │
│   (GitHub Actions: src/training_pipeline.py)            │
└───────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────┐
│   Hopsworks Model    │
│       Registry       │
└───────────┬──────────┘
│ (Best Model Artifacts)
▼
┌─────────────────────────────────────────────────────────┐
│               Streamlit Web Application                 │
│   (Live Infiltration & Prediction Dashboard)            │
└─────────────────────────────────────────────────────────┘

```

1. **Feature Pipeline (Hourly):** 
   - Cron-triggered GitHub Action fetches live hourly weather and particulate matter ($PM_{2.5}$, $PM_{10}$, $NO_2$, $CO$, etc.) data.
   - Cleans, transforms, and uploads enriched feature sets to the Hopsworks Feature Store.
2. **Training Pipeline (Daily):** 
   - Automated pipeline retrieves fresh historical feature frames from Hopsworks.
   - Retrains the machine learning model, evaluates metrics against baseline, and updates the Hopsworks Model Registry.
3. **Inference Pipeline (Real-Time):**
   - Streamlit dashboard fetches live feature groups and model artifacts to display current AQI status and future predictions.

---

## ✨ Key Features

* **🤖 Fully Automated MLOps:** Scheduled backend automation using GitHub Actions runner environments.
* **📦 Centralized Feature Management:** Seamless integration with **Hopsworks Feature Store** for zero data drift between training and inference.
* **📈 Dynamic Visualizations:** Interactive Plotly dashboards embedded in Streamlit showing trends, historical patterns, and forecasts.
* **🛡️ Continuous Deployment:** Auto-syncing production pipeline with robust error handling and API key environment isolation.

---

## 🛠️ Tech Stack

* **Language:** Python 3.12
* **Data Manipulation & Analytics:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, TensorFlow / Joblib
* **Data Visualization:** Plotly Express, Streamlit
* **Feature Store & Model Registry:** Hopsworks (`confluent-kafka`, `hopsworks-sdk`)
* **CI/CD & Orchestration:** GitHub Actions
* **Data Provider:** AQICN API

---

## 📂 Repository Structure

```text
aqi-predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml     # Hourly data ingestion & feature upload
│       └── training_pipeline.yml    # Daily model retraining pipeline
├── src/
│   ├── data/
│   │   └── data_cleaning.py         # Data preprocessing utilities
│   ├── feature_pipeline.py          # API fetching logic
│   ├── uploading_to_hopsworks.py    # Feature Store upload handler
│   └── training_pipeline.py         # Model training & registry update logic
├── app.py                           # Streamlit dashboard application
├── requirements.txt                 # Project dependencies
├── .gitignore
└── README.md                        # Documentation

```

---

## 🚀 Local Setup & Installation

Follow these steps to run the project locally on your machine:

### 1. Clone the Repository

```bash
git clone [https://github.com/mjamshaid786/aqi-predictor.git](https://github.com/mjamshaid786/aqi-predictor.git)
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

Create a `.env` file in the root directory and insert your credentials:

```ini
AQICN_API_KEY=your_aqicn_api_key_here
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT=aqi_predictor_2026
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai

```

### 5. Run Streamlit Dashboard

```bash
streamlit run app.py

```

---

## 🔐 Environment Secrets & Configuration

To enable GitHub Actions pipelines to execute properly, configure the following secrets in **Repository Settings -> Secrets and variables -> Actions**:

| Secret Name | Description |
| --- | --- |
| `AQICN_API_KEY` | API Key obtained from AQICN for fetching raw pollution data |
| `HOPSWORKS_API_KEY` | User API token generated from Hopsworks Cloud |
| `HOPSWORKS_PROJECT` | Active Hopsworks Project identifier |



## 🤝 Author & Acknowledgments

* **Developer:** Muhammad Jamshaid
* **GitHub:** [@mjamshaid786](https://www.google.com/search?q=https://github.com/mjamshaid786)
* **Special Thanks:** Built as part of an advanced hands-on MLOps implementation utilizing open-source infrastructure.

```
