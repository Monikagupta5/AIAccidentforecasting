# 🚦 AI Accident Prediction

## 📋 Description
This project is part of the **AI Engineering Challenge** by Digital Product School. It aims to forecast the number of monthly accidents for the category **"Alkoholunfälle" (Alcohol-related accidents)** using historical data from the München Open Data Portal. The dataset includes monthly statistics on various types of accidents up to the end of 2020.

The solution involves:
- **Data preprocessing** to handle missing values and filter relevant columns.
- Building a **time-series forecasting model** using ARIMA.
- Exposing the model predictions via a RESTful API deployed on a cloud platform.

---

## 🎯 Objectives
1. Visualize historical accident data by category.
2. Forecast the number of alcohol-related accidents (`Alkoholunfälle`) for January 2021.
3. Deploy the solution and expose predictions through an API.

---

## 📊 Dataset
The dataset, sourced from the [München Open Data Portal](https://opendata.muenchen.de/), includes the following columns:
- **Category**: The accident category (e.g., Alkoholunfälle).
- **Accident-type**: Specifies whether it's a total or subcategory.
- **Year**: The year of the data.
- **Month**: The month of the data.
- **Value**: The number of accidents.

---

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas` for data manipulation
  - `matplotlib` for visualization
  - `statsmodels` and `pmdarima` for ARIMA modeling
  - `Flask` for API development
- **Deployment Platform**: Heroku (or any other cloud service)

---

## ⚙️ How to Run the Project

### Prerequisites
1. Install Python 3.8 or above.
2. Clone this repository:
   ```bash
   git clone https://github.com/monikagupta5/AIAccidentPrediction.git
   cd AIAccidentPrediction
