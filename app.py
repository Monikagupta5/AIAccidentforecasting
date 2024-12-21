from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()

# Load the pre-processed and cleaned data
try:
    df = pd.read_csv('cleaned_data.csv', parse_dates=['Date'], index_col='Date')
except FileNotFoundError:
    raise FileNotFoundError("The cleaned data file 'cleaned_data.csv' is missing.")

# Pre-trained ARIMA model
order = (2, 1, 1)  # Replace with the actual order from your Auto-ARIMA step
model = ARIMA(df['Value'], order=order).fit()

@app.post("/predict")
async def predict(year: int, month: int):
    """
    Predict accident numbers for a given year and month.

    Input:
    - year: int (e.g., 2021)
    - month: int (1-12)

    Output:
    - {"prediction": value}
    """
    # Validate the input month
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Invalid month. Must be between 1 and 12.")
    
    # Create the index for forecasting
    date_index = pd.date_range(f"{year}-{month:02d}-01", periods=1, freq='MS')
    
    # Forecast the value using the ARIMA model
    forecast = model.forecast(steps=len(date_index))
    
    return {"prediction": float(forecast[0])}
