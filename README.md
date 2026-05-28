# CVD XGBoost Streamlit Prototype

This Streamlit app uses a reduced XGBoost model pipeline to return an estimated cardiovascular disease risk percentage.

The web form uses height and weight to calculate BMI automatically. The calculated BMI is then passed into the trained model pipeline.

Required files:

- `app.py`
- `requirements.txt`
- `runtime.txt`
- `reduced_xgb_pipeline.pkl`
- `model_metadata.pkl`
