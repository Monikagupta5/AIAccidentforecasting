from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import os
import os

app = FastAPI(
    title="AI Accident Prediction API",
    description="An API to predict accident numbers using a pre-trained ARIMA model.",
    version="1.0.0"
)

# Path to the data file
DATA_FILE = os.getenv("DATA_FILE", "cleaned_data.csv")
app = FastAPI(
    title="AI Accident Prediction Web App",
    description="A web app to predict accident numbers using a pre-trained ARIMA model.",
    version="1.0.0"
)

# Set up templates directory
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the pre-processed and cleaned data
df = None
DATA_FILE = os.getenv("DATA_FILE", "cleaned_data.csv")
df = None
try:
    df = pd.read_csv(DATA_FILE, parse_dates=['Date'], index_col='Date')
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")
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
    print(f"Error: The file '{DATA_FILE}' is missing.")
    df = None

# ARIMA Model Configuration
ORDER = (2, 1, 1)  # Replace with your actual order
model = None
if df is not None:
    try:
        model = ARIMA(df["Value"], order=ORDER).fit()
    except Exception as e:
        print(f"Error training ARIMA model: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the home page with a prediction form.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, year: int = Form(...), month: int = Form(...)):
    """
    Handle form submission and display the prediction result.
    """
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
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid month. Must be between 1 and 12."})
    if year < 2000 or year > 2100:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid year. Must be between 2000 and 2100."})

    if model is None:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Model is not loaded. Cannot make predictions."})

    try:
        date_index = pd.date_range(f"{year}-{month:02d}-01", periods=1, freq="MS")
        forecast = model.forecast(steps=len(date_index))
        prediction = float(forecast[0])
        return templates.TemplateResponse("index.html", {"request": request, "result": f"The predicted value for {year}-{month:02d} is {prediction:.2f}"})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Prediction failed: {str(e)}"})
