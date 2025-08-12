import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Care Cost Predictor", layout="wide")

# ---------- Paths & Data ----------
DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "insurance_cleaned.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Load models (pretrained)
@st.cache_resource
def load_models():
    models = {}
    try:
        models["XGBoost"] = joblib.load(MODELS_DIR / "xgb_model.pkl")
    except Exception as e:
        st.warning(f"Could not load XGBoost model: {e}")
    try:
        models["Random Forest"] = joblib.load(MODELS_DIR / "rf_model.pkl")
    except Exception as e:
        st.warning(f"Could not load Random Forest model: {e}")
    try:
        models["Linear Regression"] = joblib.load(MODELS_DIR / "lr_model.pkl")
    except Exception as e:
        st.warning(f"Could not load Linear Regression model: {e}")
    return models

@st.cache_resource
def load_background():
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        return df
    return None

models = load_models()
bg_df_full = load_background()

# Compute dataset average charge (for comparison)
average_charge = None
if bg_df_full is not None and "charges" in bg_df_full.columns:
    average_charge = float(bg_df_full["charges"].mean())

# ---------- Helpers ----------
def make_model_input(age, bmi, sex, children, smoker, region):
    # Transform raw inputs into one-hot encoded row expected by the models.
    return pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
    })

def predict_with(model_name, X):
    mdl = models.get(model_name)
    if mdl is None:
        raise RuntimeError(f"Model '{model_name}' is not available.")
    return float(mdl.predict(X)[0])

# ---------- UI: Tabs ----------
tab1, tab2, tab3 = st.tabs(["🧮 Predict Cost", "🧪 What‑If Analysis", "🧠 Explain Model"])

# ========== Tab 1: Single Prediction ==========
with tab1:
    st.header("Predict Cost")
    st.caption("Estimate annual medical charges for a single profile.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 100, 30, help="Age in years.")
        bmi = st.slider("BMI", 10.0, 50.0, 25.0, step=0.1, help="Body Mass Index (kg/m²).")
    with col2:
        sex = st.selectbox("Sex", ["male", "female"], help="Biological sex.")
        children = st.slider("Children", 0, 5, 0, help="Number of dependents.")
    with col3:
        smoker = st.selectbox("Smoker", ["yes", "no"], help="Smoking status.")
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], help="US region of residence.")

    model_choice = st.selectbox("Model", ["XGBoost", "Random Forest", "Linear Regression"], help="Pick a trained model for prediction.")

    X = make_model_input(age, bmi, sex, children, smoker, region)

    if st.button("Predict"):
        try:
            yhat = predict_with(model_choice, X)
            st.success(f"Estimated Medical Charges: **${yhat:,.2f}**")

            if average_charge is not None:
                diff = yhat - average_charge
                pct = (diff / average_charge) * 100
                if diff > 0:
                    st.info(f"Your predicted charge is **{pct:.1f}% higher** than the dataset average (${average_charge:,.2f}).")
                elif diff < 0:
                    st.info(f"Your predicted charge is **{abs(pct):.1f}% lower** than the dataset average (${average_charge:,.2f}).")
                else:
                    st.info("Your predicted charge is approximately the same as the dataset average.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # --- SHAP waterfall (commented by request) ---
    # with st.expander("SHAP Waterfall (disabled)", expanded=False):
    #     st.caption("This visualization is intentionally disabled. All SHAP content moved to the 'Explain Model' tab.")
    #     # Old code example (kept for future reference):
    #     # explainer = shap.Explainer(models['XGBoost'], bg_df_full.drop(columns=['charges']).sample(100, random_state=42))
    #     # shap_values = explainer(X)
    #     # shap.plots.waterfall(shap_values[0])

# ========== Tab 2: Scenario Analysis ==========
with tab2:
    st.header("What‑If Analysis")
    st.caption("Create multiple profiles in a data editor and compare predicted costs side‑by‑side (no data is persisted).")

    # Provide a starter table that users can edit (add rows via UI)
    starter = pd.DataFrame([{
        "age": 30,
        "bmi": 25.0,
        "sex": "male",
        "children": 0,
        "smoker": "no",
        "region": "northeast",
    }])
    st.markdown("**Edit the table below, add rows, and then click *Run Comparison*.**")
    edited = st.data_editor(
        starter,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "age": st.column_config.NumberColumn("Age", min_value=18, max_value=100, help="Age in years."),
            "bmi": st.column_config.NumberColumn("BMI", min_value=10.0, max_value=50.0, step=0.1, help="Body Mass Index (kg/m²)."),
            "sex": st.column_config.SelectboxColumn("Sex", options=["male", "female"], help="Biological sex."),
            "children": st.column_config.NumberColumn("Children", min_value=0, max_value=5, step=1, help="Number of dependents."),
            "smoker": st.column_config.SelectboxColumn("Smoker", options=["yes", "no"], help="Smoking status."),
            "region": st.column_config.SelectboxColumn("Region", options=["northeast", "northwest", "southeast", "southwest"], help="US region."),
        }
    )

    model_choice2 = st.selectbox("Model", ["XGBoost", "Random Forest", "Linear Regression"], key="scenario_model", help="Pick a trained model for comparison.")

    if st.button("Run Comparison"):
        try:
            preds = []
            for _, row in edited.iterrows():
                Xi = make_model_input(
                    int(row["age"]), float(row["bmi"]), str(row["sex"]),
                    int(row["children"]), str(row["smoker"]), str(row["region"])
                )
                yhat = predict_with(model_choice2, Xi)
                preds.append(yhat)
            result = edited.copy()
            result["predicted_charges"] = np.round(preds, 2)
            st.subheader("Predictions")
            st.dataframe(result, use_container_width=True)

            # Plotly bar chart
            fig = px.bar(
                result.reset_index().rename(columns={"index": "Scenario"}),
                x="Scenario", y="predicted_charges",
                title="Predicted Medical Charges by Scenario",
                labels={"predicted_charges": "Charges ($)"},
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Scenario comparison failed: {e}")

# ========== Tab 3: Model Explainability ==========
with tab3:
    st.header("Explain Model")
    st.caption("Interactive SHAP dashboard: adjust background sample size, choose model and feature for dependence plot.")

    if bg_df_full is None:
        st.warning("Background dataset not found at `data/insurance_cleaned.csv`. Please add it to enable SHAP.")
    else:
        cols = bg_df_full.columns.tolist()
        if "charges" in cols:
            feature_df = bg_df_full.drop(columns=["charges"])
        else:
            feature_df = bg_df_full.copy()

        model_choice3 = st.selectbox("Model", ["XGBoost", "Random Forest", "Linear Regression"], key="expl_model", help="Model to explain with SHAP.")
        sample_size = st.slider("Background sample size", 50, min(1000, len(feature_df)), min(300, len(feature_df)), step=50,
                                help="Fewer rows = faster, more rows = more stable explanations.")

        # Sample background
        bg_sample = feature_df.sample(sample_size, random_state=42) if len(feature_df) > sample_size else feature_df

        # Explainer & shap values
        try:
            explainer = shap.Explainer(models[model_choice3], bg_sample)
            shap_values = explainer(bg_sample)

            st.subheader("Feature Importance (mean |SHAP|)")
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            imp_df = pd.DataFrame({"feature": bg_sample.columns, "importance": mean_abs}).sort_values("importance", ascending=False)
            fig_imp = px.bar(imp_df, x="feature", y="importance", title="Mean |SHAP| Feature Importance", template="plotly_dark")
            fig_imp.update_layout(xaxis_title="", yaxis_title="Mean |SHAP|")
            st.plotly_chart(fig_imp, use_container_width=True)

            st.subheader("Dependence Plot")
            feat = st.selectbox("Feature", list(bg_sample.columns), help="Feature to visualize vs. its SHAP value.")
            fig_dep = px.scatter(
                x=bg_sample[feat],
                y=shap_values.values[:, list(bg_sample.columns).index(feat)],
                color=bg_sample[feat],
                labels={"x": feat, "y": "SHAP value"},
                title=f"SHAP Dependence: {feat}",
                template="plotly_dark"
            )
            st.plotly_chart(fig_dep, use_container_width=True)
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")