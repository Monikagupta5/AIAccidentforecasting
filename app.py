from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import os

app = FastAPI(
    title="AI Accident Prediction API",
    description="An API to predict accident numbers using a pre-trained ARIMA model.",
    version="1.0.0"
)

# Path to the data file
DATA_FILE = os.getenv("DATA_FILE", "cleaned_data.csv")

# Load the pre-processed and cleaned data
df = None
try:
    df = pd.read_csv(DATA_FILE, parse_dates=['Date'], index_col='Date')
except FileNotFoundError:
    print(f"Error: The file '{DATA_FILE}' is missing.")
    df = None  # Proceed without crashing

# ARIMA Model Configuration
ORDER = (2, 1, 1)  # Replace with your actual order
model = None
if df is not None:
    try:
        model = ARIMA(df['Value'], order=ORDER).fit()
    except Exception as e:
        print(f"Error training ARIMA model: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the AI Accident Prediction API!"}

@app.get("/health")
async def health_check():
    """Check if the app is running and model is ready."""
    if model is None or df is None:
        return {"status": "error", "message": "Model or data is not ready."}
    return {"status": "ok", "message": "App is running and model is ready."}

@app.post("/predict")
async def predict(year: int, month: int):
    """
    Predict accident numbers for a given year and month.

    Parameters:
    - year: int (e.g., 2021)
    - month: int (1-12)

    Returns:
    - {"prediction": value}
    """
    # Validate inputs
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Invalid month. Must be between 1 and 12.")
    if year < 2000 or year > 2100:  # Example range for valid years
        raise HTTPException(status_code=400, detail="Invalid year. Must be between 2000 and 2100.")

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Cannot make predictions.")

    # Create the index for forecasting
    try:
        date_index = pd.date_range(f"{year}-{month:02d}-01", periods=1, freq='MS')
        forecast = model.forecast(steps=len(date_index))
        return {"prediction": float(forecast[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
