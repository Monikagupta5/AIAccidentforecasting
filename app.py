import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load the pre-processed and cleaned data
@st.cache
def load_data():
    try:
        return pd.read_csv("cleaned_data.csv", parse_dates=['Date'], index_col='Date')
    except FileNotFoundError:
        st.error("The file 'cleaned_data.csv' is missing.")
        return None

df = load_data()

# ARIMA Model Configuration
@st.cache(allow_output_mutation=True)
def train_model(data):
    """
    Train the ARIMA model based on the loaded dataset.
    """
    try:
        order = (2, 1, 1)  # Replace with your actual ARIMA order
        return ARIMA(data['Value'], order=order).fit()
    except Exception as e:
        st.error(f"Error training ARIMA model: {e}")
        return None

model = train_model(df) if df is not None else None

# Streamlit App Layout
st.title("AI Accident Prediction")
st.write("Predict accident numbers using a pre-trained ARIMA model.")

# Sidebar for inputs
st.sidebar.header("Prediction Input")
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2023)
month = st.sidebar.number_input("Month (1-12)", min_value=1, max_value=12, value=1)

# Check if the model is ready
if model is None:
    st.error("Model is not loaded. Ensure 'cleaned_data.csv' is present and the model can be trained.")
else:
    # Prediction button
    if st.sidebar.button("Predict"):
        try:
            # Validate inputs
            if not (1 <= month <= 12):
                st.error("Invalid month. Must be between 1 and 12.")
            else:
                # Generate prediction
                date_index = pd.date_range(f"{year}-{month:02d}-01", periods=1, freq='MS')
                forecast = model.forecast(steps=len(date_index))
                st.success(f"Prediction for {year}-{month:02d}: {forecast[0]:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
