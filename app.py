import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd

# Load the model from MLflow
model_name = "XGB"
model_version = "1"  # Update to the latest version from MLflow UI
model_uri = f"models:/{model_name}/{model_version}"
try:
    model = mlflow.sklearn.load_model(model_uri)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Streamlit UI
st.title("XGBoost Sales Forecasting")
st.write("Enter the features to predict the next day's sales.")

# Feature inputs (replace with actual feature names from X_train)
feature_names = ["transactions", "dcoilwtico", "holiday_type", "is_weekend", "month", "day_of_week", "day_of_month", "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30"]  # Adjust based on X_train columns
input_data = {}
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0, step=10.0)
    input_data[feature] = value

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data], columns=feature_names)
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Sales: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

if st.button("Clear"):
    for feature in feature_names:
        st.session_state[f"{feature}"] = 0.0
    st.experimental_rerun()