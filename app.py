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

DIABETES_OPTIONS = [
    "No",
    "No, pre-diabetes or borderline diabetes",
    "Yes",
    "Yes, but female told only during pregnancy",
]


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
    return "YES" if risk_score >= metadata["threshold"] else "NO"


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
      .result-yes {
        border-color: rgba(199, 85, 67, 0.45);
        background: #fff1ee;
      }
      .result-yes strong {
        color: #c75543;
      }
      .result-no {
        border-color: rgba(25, 116, 71, 0.4);
        background: #eef8f2;
      }
      .result-no strong {
        color: #197447;
      }
      .note {
        border-left: 4px solid #b57924;
        background: #fff8ec;
        padding: 12px 14px;
        border-radius: 6px;
        color: #5d461d;
        margin-top: 16px;
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
      <p>XGBoost model - reduced 8-feature pipeline - binary output only</p>
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

    bmi = st.slider("BMI", min_value=12.0, max_value=60.0, value=25.0, step=0.1)

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
        result = predict(input_data)
        result_class = "result-yes" if result == "YES" else "result-no"
        result_text = (
            "The model predicts heart disease for this input profile."
            if result == "YES"
            else "The model does not predict heart disease for this input profile."
        )
        st.markdown(
            f"""
            <div class="result-card {result_class}">
              <span>Heart disease prediction</span>
              <strong>{result}</strong>
              <p>{result_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card">
              <span>Heart disease prediction</span>
              <strong style="font-size: 34px; color: #607078;">Waiting</strong>
              <p>Enter the patient features and run the model.</p>
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
