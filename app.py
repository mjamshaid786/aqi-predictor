"""
Lahore Air Quality Index (AQI) Prediction System
End-to-End Serverless ML Architecture:
Hopsworks Feature Store + Model Registry + Streamlit Dashboard
"""
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import hopsworks
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensorflow as tf
from inference_pipeline import generate_3day_forecast

# ----------------------------------------------------------------------------
# 1. Page Config & Custom Styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Lahore AQI Forecast | Serverless ML",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished metric cards & badges
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222B;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4CAF50;
    }
    .badge-info {
        background-color: #262730;
        padding: 6px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        color: #00E676;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
FG_NAME, FG_VERSION = "aqi_predictions", 7

CANDIDATE_MODEL_NAMES = [
    "aqi_neural_network_model",
    "aqi_lasso_model",
    "aqi_gradient_boosting_model",
    "aqi_ridge_model",
    "aqi_random_forest_model",
]
DROP_COLS = ["date", "city", "aqi_target"]

# ----------------------------------------------------------------------------
# 2. Hopsworks Connection & Cached Loaders
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Hopsworks Feature Store...")
def get_project():
    return hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT", "aqi_predictor_2026"),
        host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        port=443,
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )


# ----------------------------------------------------------------------------
# Fixed & Updated load_model() Function
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_registry():
    return get_project().get_model_registry()


@st.cache_resource(ttl=3600, show_spinner="Downloading best model from Hopsworks Model Registry...")
def load_model():
    project = get_project()
    mr = project.get_model_registry()

    candidates = []
    for name in CANDIDATE_MODEL_NAMES:
        try:
            candidates.extend(mr.get_models(name))
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            f"None of the configured model names were found: {CANDIDATE_MODEL_NAMES}."
        )

    # ----------------------------------------------------------------------------
    # 🎯 LATEST BATCH BEST MODEL SELECTION LOGIC
    # ----------------------------------------------------------------------------
    # 1. Sab se pehle Registry se Max (Latest) Version Number nikalein
    max_version = max([int(m.version) for m in candidates if str(m.version).isdigit()], default=1)

    # 2. Sirf LATEST VERSION wale models filter karein (purane versions skip ho jayenge)
    latest_candidates = [m for m in candidates if int(m.version) == max_version]

    # 3. RMSE metrics safely extract karne ka function
    def get_rmse(m):
        if not m.training_metrics:
            return float("inf")
        metrics = m.training_metrics
        for k in ["test_rmse", "rmse", "RMSE"]:
            if k in metrics:
                try:
                    return float(metrics[k])
                except (ValueError, TypeError):
                    pass
        return float("inf")

    # 4. Latest version wale models ko lowest RMSE ke hisab se sort karein
    ordered = sorted(latest_candidates, key=get_rmse)
    # ----------------------------------------------------------------------------

    errors = []
    for model_meta in ordered:
        try:
            model_dir = model_meta.download()
        except Exception as e:
            errors.append(f"{model_meta.name} v{model_meta.version}: download failed ({e})")
            continue

        loaded = None
        for fname in ("model.pkl", "model.joblib"):
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                loaded = joblib.load(fpath)
                break

        if loaded is None:
            h5_path = os.path.join(model_dir, "model.h5")
            if os.path.exists(h5_path):
                try:
                    loaded = tf.keras.models.load_model(h5_path, compile=False)
                except Exception as e:
                    errors.append(f"{model_meta.name} v{model_meta.version}: model.h5 load failed ({e})")
                    continue

        if loaded is not None:
            feature_names = None
            feat_path = os.path.join(model_dir, "feature_names.txt")
            if os.path.exists(feat_path):
                with open(feat_path) as f:
                    feature_names = [line.strip() for line in f if line.strip()]

            scaler = None
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)

            return loaded, scaler, feature_names, model_meta.name, model_meta.version, model_meta.training_metrics

        errors.append(f"{model_meta.name} v{model_meta.version}: missing weights")

    raise RuntimeError("No usable model found:\n" + "\n".join(errors))


@st.cache_data(ttl=300, show_spinner="Fetching latest feature pipeline records...")
def load_features():
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    df = fg.read()
    sort_col = "timestamp" if "timestamp" in df.columns else "date"
    return df.sort_values(sort_col).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. Inference & Forecasting Logic
# ----------------------------------------------------------------------------
def pm25_to_aqi(pm25: float) -> float:
    """
    Convert a raw PM2.5 concentration (µg/m³) into the actual US EPA AQI
    index (0-500 scale) using the official piecewise-linear breakpoint
    formula. The model predicts raw PM2.5 concentration (that's what
    'aqi_target' is, despite the name) -- this converts it to the real
    AQI number so it matches what IQAir/AQICN/etc. report, instead of
    displaying the raw µg/m³ value mislabeled as "AQI".
    """
    if pm25 is None or pd.isna(pm25):
        return float("nan")
    pm25 = max(0.0, float(pm25))
    # (C_low, C_high, I_low, I_high)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return (i_high - i_low) / (c_high - c_low) * (pm25 - c_low) + i_low
    return 500.0  # above scale -- cap at max


def predict(model, scaler, feature_names, df: pd.DataFrame) -> pd.Series:
    if feature_names:
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Feature store missing required columns: {missing}")
        X = df[feature_names].copy()
    else:
        X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    if scaler is not None:
        X = scaler.transform(X)

    preds = np.asarray(model.predict(X)).reshape(-1)
    return pd.Series(preds, index=df.index)


@st.cache_data(ttl=3600, show_spinner="Computing SHAP explanations (this can take a minute)...")
def compute_shap_values(_model, _scaler, feature_names, df_features: pd.DataFrame,
                         background_size: int = 50, explain_size: int = 30):
    """
    Model-agnostic SHAP explanation: works identically for tree models
    (Random Forest, Gradient Boosting), linear models (Ridge, Lasso), and
    the Keras neural network, since it wraps model.predict() directly
    rather than needing a model-specific SHAP explainer.
    Cached by Streamlit (leading underscore on _model/_scaler tells
    Streamlit not to try hashing the unhashable model object itself).
    """
    import shap

    X = df_features[feature_names].copy()
    background = X.sample(n=min(background_size, len(X)), random_state=42)
    explain_sample = X.sample(n=min(explain_size, len(X)), random_state=7)

    def predict_fn(data):
        arr = np.asarray(data)
        if _scaler is not None:
            arr = _scaler.transform(arr)
        return np.asarray(_model.predict(arr)).reshape(-1)

    explainer = shap.Explainer(predict_fn, background)
    shap_values = explainer(explain_sample)
    return shap_values, explain_sample


def render_alert_banner(aqi: float):
    if aqi > 200:
        st.error(f"🚨 **VERY HAZARDOUS AQI ({aqi:.0f})** — Serious health effects for all populations. Avoid all outdoor activities!")
    elif aqi > 150:
        st.error(f"🚨 **HAZARDOUS AQI ({aqi:.0f})** — Unhealthy for general public. Wear N95 masks outdoors.")
    elif aqi > 100:
        st.warning(f"⚠️ **UNHEALTHY FOR SENSITIVE GROUPS ({aqi:.0f})** — Children and seniors should limit outdoor effort.")
    elif aqi > 50:
        st.info(f"🟡 **MODERATE AIR QUALITY ({aqi:.0f})** — Acceptable quality; slight concern for sensitive individuals.")
    else:
        st.success(f"✅ **GOOD AIR QUALITY ({aqi:.0f})** — Air pollution poses little or no risk.")





# ----------------------------------------------------------------------------
# 4. Streamlit UI Layout
# ----------------------------------------------------------------------------
# Sidebar Info
st.sidebar.title("🛠️ System Overview")
st.sidebar.markdown("**Architecture:** 100% Serverless ML Stack")
st.sidebar.markdown("- **Feature Store:** Hopsworks")
st.sidebar.markdown("- **Model Registry:** Hopsworks")
st.sidebar.markdown("- **Pipeline Automation:** GitHub Actions")
st.sidebar.markdown("- **UI/Inference:** Streamlit")
st.sidebar.divider()

model, scaler, feature_names = None, None, None
model_name, model_version, model_metrics = None, None, {}
df = pd.DataFrame()

try:
    model, scaler, feature_names, model_name, model_version, model_metrics = load_model()
    df = load_features()
except Exception as e:
    st.error(f"Failed to load Hopsworks assets: {e}")

if model is None or df.empty:
    st.error("Cannot proceed: Hopsworks assets unavailable. Check logs.")
else:
    # Top Banner
    st.title("🌫️ Lahore Air Quality Index (AQI) Predictor")
    st.markdown("Real-time air pollution forecasting using machine learning pipelines connected to Hopsworks Feature Store.")

    # Model Performance Tag
    def fmt_metric(v):
        try:
            return f"{float(v):.2f}"
        except (TypeError, ValueError):
            return "N/A"
    metrics_str = (
        f"Test RMSE: {fmt_metric(model_metrics.get('test_rmse'))} | "
        f"R²: {fmt_metric(model_metrics.get('test_r2'))}"
    ) if isinstance(model_metrics, dict) else ""
    st.sidebar.markdown(f"🏆 **Active Model:** `{model_name}` (v{model_version})")
    if metrics_str:
        st.sidebar.caption(metrics_str)

    # Compute historical predictions
    df["predicted_aqi"] = predict(model, scaler, feature_names, df)
    
    latest = df.iloc[-1]
    current_pm25 = latest.get("pm25_avg", latest.get("pm2_5", latest.get("pm25", np.nan)))
    if pd.isna(current_pm25):
        current_pm25 = None
    predicted_aqi = pm25_to_aqi(latest["predicted_aqi"])  # convert raw PM2.5 prediction -> real AQI
    prev_actual_raw = df["aqi_target"].iloc[-2] if "aqi_target" in df.columns and len(df) > 1 else None
    prev_actual = pm25_to_aqi(prev_actual_raw) if prev_actual_raw is not None else predicted_aqi
    delta = (predicted_aqi - prev_actual) if pd.notna(prev_actual) and pd.notna(predicted_aqi) else None

    # Create Main Tabs matching PDF Requirements
    tab_forecast, tab_analytics, tab_data = st.tabs([
        "🔮 3-Day Forecast & Live AQI", 
        "📊 Model Analytics & SHAP Explanations", 
        "🗄️ Feature Store Data"
    ])

    # ------------------------------------------------------------------------
    # TAB 1: Live AQI & 3-Day Forecast
    # ------------------------------------------------------------------------
    with tab_forecast:
        # KPI Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current PM2.5 Level", f"{current_pm25:.1f} µg/m³" if current_pm25 is not None else "N/A")
        c2.metric("Next-Hour Predicted AQI", f"{predicted_aqi:.1f}")
        c3.metric("Change vs Previous Hour", f"{delta:+.1f}" if delta is not None else "N/A")
        c4.metric("Active Model", model_name.replace("aqi_", "").replace("_model", "").upper())

        st.markdown("---")
        render_alert_banner(predicted_aqi)
        st.markdown("---")

        # 3-Day Forecast Horizon Section (REAL model predictions -- one
        # genuine prediction per horizon from that horizon's own trained
        # model, not a simulated/random curve)
        st.subheader("📅 Next 3 Days AQI Forecast (Real Model Predictions)")
        forecast_df = generate_3day_forecast(df, get_registry())

        fc_col1, fc_col2, fc_col3 = st.columns(3)
        for col, (_, row) in zip([fc_col1, fc_col2, fc_col3], forecast_df.iterrows()):
            with col:
                if row["predicted_aqi"] is not None:
                    card_aqi = pm25_to_aqi(row["predicted_aqi"])
                    st.info(
                        f"**{row['label']}**\n\n### AQI: {card_aqi:.0f}\n\n"
                        f"_{row['model_name']} (v{row['model_version']})_"
                    )
                else:
                    st.warning(f"**{row['label']}**\n\nUnavailable: {row.get('error', 'model not found')}")

        # Interactive Forecast Chart
        st.subheader("📈 Historical Trend + 3-Day Forecast Points")
        
        recent = df.tail(48)
        fig = go.Figure()
        x_col = "timestamp" if "timestamp" in recent.columns else "date"
        pm_col = "pm25_avg" if "pm25_avg" in recent.columns else ("pm2_5" if "pm2_5" in recent.columns else "pm25")

        # Actual AQI line (converted from raw PM2.5 concentration, same
        # formula as everywhere else, so this is on the same scale as the
        # prediction traces below -- mixing raw µg/m³ with AQI index
        # values on one chart would be misleading)
        if pm_col in recent.columns:
            fig.add_trace(go.Scatter(
                x=recent[x_col], y=recent[pm_col].apply(pm25_to_aqi),
                name="Actual AQI", mode="lines+markers", line=dict(color="#FF9800", width=2)
            ))

        # Model Historical Predictions
        fig.add_trace(go.Scatter(
            x=recent[x_col], y=recent["predicted_aqi"].apply(pm25_to_aqi),
            name="Model Prediction (Historical)", mode="lines", line=dict(color="#00E676", width=2, dash="dash")
        ))

        # 3-Day Forecast: 3 real points (current value + day1 + day2 + day3),
        # connected with straight lines for visual continuity. These are
        # NOT hourly-resolution predictions -- each is a genuine direct
        # prediction from that horizon's ow
        valid_forecast = forecast_df.dropna(subset=["predicted_aqi"])
        forecast_x = [latest.get("timestamp", recent[x_col].iloc[-1])] + list(valid_forecast["forecast_time"])
        forecast_y = [predicted_aqi] + [pm25_to_aqi(v) for v in valid_forecast["predicted_aqi"]]
        fig.add_trace(go.Scatter(
            x=forecast_x, y=forecast_y,
            name="3-Day Forecast (Day1/Day2/Day3)", mode="lines+markers",
            line=dict(color="#00B0FF", width=2), marker=dict(size=10),
        ))

        fig.update_layout(
            xaxis_title="Timeline",
            yaxis_title="AQI (US EPA Index)",
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------------
    # TAB 2: Advanced Analytics & Feature Importance
    # ------------------------------------------------------------------------
    with tab_analytics:
        st.subheader("🔍 Model Explainability (SHAP)")
        st.caption(
            "Real SHAP (SHapley Additive exPlanations) values — computed the same way "
            "regardless of which algorithm is active (tree, linear, or neural network), "
            "since it treats the model as a black box via model.predict()."
        )

        if feature_names:
            try:
                shap_values, explain_sample = compute_shap_values(model, scaler, feature_names, df)

                # Mean |SHAP value| per feature -- overall importance ranking
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                shap_df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP value|": mean_abs_shap})
                shap_df = shap_df.sort_values(by="Mean |SHAP value|", ascending=True)

                fig_shap_bar = px.bar(
                    shap_df, x="Mean |SHAP value|", y="Feature", orientation="h",
                    title="Feature Importance (Mean Absolute SHAP Value)",
                    color="Mean |SHAP value|", color_continuous_scale="Viridis"
                )
                fig_shap_bar.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_shap_bar, use_container_width=True)

                # True SHAP beeswarm plot -- shows both magnitude AND direction
                # (does a high value of this feature push the prediction up or down)
                st.markdown("**SHAP Summary (Beeswarm) Plot**")
                st.caption("Each dot is one prediction. Color = feature value (red=high, blue=low). "
                           "Position = impact on that prediction (right = pushes AQI up).")
                import matplotlib.pyplot as plt
                import shap as shap_lib
                fig_beeswarm = plt.figure()
                shap_lib.plots.beeswarm(shap_values, show=False)
                st.pyplot(fig_beeswarm, use_container_width=True)
                plt.close(fig_beeswarm)

            except Exception as e:
                st.warning(f"Could not compute SHAP values: {e}")
        else:
            st.info("SHAP explanations require the model's feature_names.txt (should be present for all registered models).")

        st.markdown("---")
        st.subheader("📊 Exploratory Pollution Trends (EDA)")
        col_eda1, col_eda2 = st.columns(2)

        with col_eda1:
            if "hour" in df.columns and "pm25_avg" in df.columns:
                hourly_avg = df.groupby("hour")["pm25_avg"].mean().reset_index()
                fig_hourly = px.line(
                    hourly_avg, x="hour", y="pm25_avg", 
                    title="Average PM2.5 Concentration by Hour of Day",
                    markers=True
                )
                fig_hourly.update_layout(template="plotly_dark")
                st.plotly_chart(fig_hourly, use_container_width=True)

        with col_eda2:
            num_cols = [c for c in ["pm25_avg", "pm10_avg", "o3_avg", "uvi_avg", "aqi_change_rate"] if c in df.columns]
            if len(num_cols) > 1:
                corr = df[num_cols].corr()
                fig_corr = px.imshow(
                    corr, text_auto=True, title="Pollutant Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                fig_corr.update_layout(template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------------------------------------------------------------
    # TAB 3: Raw Feature Store Inspection
    # ------------------------------------------------------------------------
    with tab_data:
        st.subheader("🗄️ Hopsworks Feature Group Explorer")
        st.caption(f"Feature Group Name: `{FG_NAME}` | Version: `{FG_VERSION}`")
        
        st.dataframe(df, use_container_width=True)
        
        # Download Data Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned Features CSV",
            data=csv,
            file_name="lahore_aqi_features.csv",
            mime="text/csv",
        )