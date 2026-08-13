# Air Quality Index (AQI) Forecasting System

An interactive web application built with Streamlit and Machine Learning to predict Air Quality Index (AQI) levels, explore global historical pollution trends, and evaluate regression model performance.

**Live Application:** [Air Quality Forecaster on Streamlit Cloud](https://mca-ml-2547242-air-quality-forecaster.streamlit.app/)

## Features

* **Real-Time AQI Predictor:** Interactive inputs for meteorological conditions and pollutant metrics (PM2.5, PM10, NO2, SO2, CO, O3) to generate instant AQI forecasts and gauge visualizations.
* **Trend & Location Analytics:** Visualizations for historical location trends, pollutant concentration breakdowns, top polluted cities comparison, and a global country-wise AQI map using Viridis color scaling.
* **Model Evaluation:** Performance dashboard comparing actual target AQI against model predictions using a scatter plot, along with MAE and RMSE metric reporting.

## Project Directory Structure

```text
cia-3/
│
├── app/
│   └── app.py                      # Main Streamlit dashboard application
├── data/
│   └── global_air_pollution_dataset.csv # Global air quality dataset
├── models/
│   └── stacking_regressor_model.pkl# Pre-trained Stacking Regressor model
├── notebook/
│   └── ml_pipeline.ipynb           # Data preprocessing and model training notebook
├── README.md                       # Project documentation
└── requirements.txt                # List of Python dependencies
```

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Installation

1. **Clone or download the repository:**
   ```bash
   git clone <repository-url>
   cd cia-3
   ```

2. **Create and activate a virtual environment (Optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the Streamlit dashboard using the following command:

```bash
streamlit run app/app.py
```

Open the displayed local URL in your web browser to access the application.

## Usage Instructions

1. **Real-Time AQI Predictor:**
   - Input desired values for temperature, humidity, wind speed, and various pollutant concentrations.
   - Select a prediction date.
   - Click **"Calculate Predicted AQI"** to view the forecasted AQI and corresponding visualization.

2. **Trend and Location Analytics:**
   - Select a location or country from the dropdown menus.
   - Adjust the year or pollutant type filters to explore historical trends.
   - Examine the displayed charts for insights into pollution levels and geographic distribution.

3. **Model Evaluation:**
   - Review the scatter plot comparing actual versus predicted AQI values.
   - Note the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics.
   - Use the reference line (y = x) to assess prediction accuracy.

## Model Details

The system uses a **Stacking Regressor** machine learning model trained on historical air quality data. Key features include:

* Input features: Comprehensive meteorological parameters, pollutant concentrations, and geographic data.
* Target variable: Air Quality Index (AQI).
* Model type: Ensemble stacking with optimized hyperparameters.
* Performance metrics: MAE, RMSE, and visualization of prediction accuracy against actual values.

## Data Source

The application utilizes historical air quality measurements stored in `data/global_air_pollution_dataset.csv`.

* **Dataset Link:** [BreathWatch Air Quality and Health (2015–2024) on Kaggle](https://www.kaggle.com/datasets/aliyasaly1231/breathwatch-air-quality-and-health-2015-2024/data)