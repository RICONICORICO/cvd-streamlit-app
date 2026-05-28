from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "reduced_xgb_pipeline.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"

AGE_OPTIONS = [
    "18-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80+",
]

GENERAL_HEALTH_OPTIONS = ["Poor", "Fair", "Good", "Very Good", "Excellent"]

CHECKUP_OPTIONS = [
    "Within the past year",
    "Within the past 2 years",
    "Within the past 5 years",
    "5 or more years ago",
    "Never",
]

DIABETES_OPTIONS = ["No", "Yes"]


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with METADATA_PATH.open("rb") as metadata_file:
        metadata = pickle.load(metadata_file)
    return model, metadata


def predict(input_data):
    model, metadata = load_model()
    input_df = pd.DataFrame([input_data], columns=metadata["features"])
    risk_score = model.predict_proba(input_df)[0, 1]
    threshold = metadata["threshold"]
    risk_percentage = risk_score * 100

    if risk_score >= threshold:
        risk_group = "Higher CVD risk"
        action_text = "This profile has a higher model-estimated cardiovascular risk."
    elif risk_score >= threshold * 0.7:
        risk_group = "Moderate CVD risk"
        action_text = "This profile is close to the model threshold and may need monitoring."
    else:
        risk_group = "Lower CVD risk"
        action_text = "This profile has a lower model-estimated cardiovascular risk."

    return risk_percentage, risk_group, action_text


def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    return weight_kg / (height_m * height_m)


st.set_page_config(
    page_title="CVD Risk Prediction",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 2rem;
        max-width: 1120px;
      }
      h1 {
        font-size: 30px !important;
        line-height: 1.15 !important;
      }
      h2, h3 {
        font-size: 18px !important;
      }
      .hero {
        border: 1px solid #dbe5e8;
        border-radius: 8px;
        padding: 18px 20px;
        background: linear-gradient(135deg, #f7fbfc 0%, #eef7f5 100%);
        margin-bottom: 18px;
      }
      .hero p {
        color: #5d6b73;
        margin-bottom: 0;
      }
      .hero h1 {
        color: #22313a;
      }
      .result-card {
        border-radius: 8px;
        padding: 24px;
        border: 1px solid #dbe5e8;
        background: #ffffff;
        min-height: 180px;
      }
      .result-card span {
        color: #607078;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
      }
      .result-card strong {
        display: block;
        font-size: 56px;
        line-height: 1;
        margin-top: 12px;
      }
      .result-risk {
        border-color: rgba(31, 101, 130, 0.35);
        background: #f1f8fb;
      }
      .result-risk strong {
        color: #1f6582;
      }
      .percentage-bar {
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: #dce9ee;
        overflow: hidden;
        margin: 18px 0 12px;
      }
      .percentage-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #4f9b73 0%, #d6a038 58%, #c75543 100%);
      }
      .note {
        border-left: 4px solid #b57924;
        background: #fff8ec;
        padding: 12px 14px;
        border-radius: 6px;
        color: #5d461d;
        margin-top: 16px;
      }
      .risk-band {
        margin-top: 14px;
        border: 1px solid #dbe5e8;
        border-radius: 8px;
        padding: 14px 16px;
        background: #f8fafb;
      }
      .risk-band span {
        display: block;
        color: #607078;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 6px;
      }
      .risk-band strong {
        display: block;
        font-size: 22px;
        line-height: 1.2;
        color: #22313a;
      }
      .risk-band p {
        margin: 8px 0 0;
        color: #5d6b73;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <p>Master of Digital Health and Data Science - Datathon Project</p>
      <h1>Cardiovascular Disease Risk Prediction</h1>
      <p>XGBoost model - reduced feature pipeline - risk percentage output</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.subheader("Patient Features")

    col1, col2 = st.columns(2)
    with col1:
        age_category = st.selectbox("Age category", AGE_OPTIONS, index=6)
        sex = st.selectbox("Sex", ["Female", "Male"], index=1)
    with col2:
        general_health = st.selectbox("General health", GENERAL_HEALTH_OPTIONS, index=2)
        checkup = st.selectbox("Last medical checkup", CHECKUP_OPTIONS, index=0)

    col_height, col_weight, col_bmi = st.columns([1, 1, 0.9])
    with col_height:
        height_cm = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=230.0,
            value=170.0,
            step=0.5,
        )
    with col_weight:
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=250.0,
            value=70.0,
            step=0.5,
        )

    bmi = round(calculate_bmi(height_cm, weight_kg), 1)
    with col_bmi:
        st.metric("Calculated BMI", f"{bmi:.1f}")

    col3, col4, col5 = st.columns(3)
    with col3:
        diabetes = st.selectbox("Diabetes", DIABETES_OPTIONS, index=0)
    with col4:
        arthritis = st.selectbox("Arthritis", ["No", "Yes"], index=0)
    with col5:
        smoking_history = st.selectbox("Smoking history", ["No", "Yes"], index=0)

    input_data = {
        "Age_Category": age_category,
        "General_Health": general_health,
        "Diabetes": diabetes,
        "Sex": sex,
        "Checkup": checkup,
        "Arthritis": arthritis,
        "BMI": bmi,
        "Smoking_History": smoking_history,
    }

    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction")

    if predict_clicked:
        risk_percentage, risk_group, action_text = predict(input_data)
        bar_width = min(max(risk_percentage, 0), 100)
        st.markdown(
            f"""
            <div class="result-card result-risk">
              <span>Estimated CVD risk</span>
              <strong>{risk_percentage:.1f}%</strong>
              <div class="percentage-bar">
                <div class="percentage-fill" style="width: {bar_width:.1f}%;"></div>
              </div>
              <p>Risk percentage calculated by the trained XGBoost model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="risk-band">
              <span>CVD risk calculation</span>
              <strong>{risk_group}</strong>
              <p>{action_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card">
              <span>Estimated CVD risk</span>
              <strong style="font-size: 34px; color: #607078;">Waiting</strong>
              <p>Enter the patient features and run the model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="risk-band">
              <span>CVD risk calculation</span>
              <strong>Waiting</strong>
              <p>The risk percentage and interpretation will appear after prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="note">
          <strong>Educational prototype only.</strong><br>
          This tool does not provide medical advice, diagnosis, or treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )
