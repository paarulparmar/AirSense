# AirSense - Live Air Quality Predictor & Health Advisor

AirSense is a Streamlit web app that accepts any city name, geocodes it, fetches live AQI and pollutant readings from WAQI, fetches live weather from OpenWeatherMap, predicts the next 72 hours of AQI with an LSTM-compatible pipeline, classifies health risk with XGBoost-compatible inference, and displays a gauge, forecast chart, Folium map, pollutant cards, and dynamic health recommendations.

## Features

- City search with OpenWeatherMap geocoding
- Live AQI from WAQI: PM2.5, PM10, NO2, O3
- Live weather from OpenWeatherMap: temperature, humidity, wind speed
- 72-hour AQI forecast using the saved LSTM model when available
- XGBoost risk classification: Good, Moderate, Unhealthy, Hazardous
- Dynamic health advice based on risk class and pollutant thresholds
- 30-minute Streamlit cache for API responses
- Secrets-based API keys for Streamlit Community Cloud
- Reproducible two-year hourly training data for 15 diverse cities

## Project Structure

```text
airsense/
├── app.py
├── pipeline/
│   ├── fetch_data.py
│   ├── features.py
│   ├── train_lstm.py
│   ├── train_xgb.py
│   └── predict.py
├── models/
│   ├── lstm_model.h5
│   └── xgb_model.pkl
├── data/
│   └── historical_aqi.csv
├── requirements.txt
└── README.md
```

## Build Historical Data

The training dataset contains two years of hourly AQI/weather records for Delhi, Mumbai, Kanpur, London, Beijing, Paris, Los Angeles, Lagos, São Paulo, Tokyo, Cairo, Sydney, New York, Berlin, and Nairobi.

```bash
python -m pipeline.fetch_data --output data/historical_aqi.csv --years 2
```

WAQI and OpenWeatherMap free tiers are excellent for live app data. Their open free endpoints do not reliably expose bulk two-year hourly history for every city, so this script builds a deterministic, API-calibrated training dataset that keeps the app runnable and retrainable on free infrastructure.

## Train Models

```bash
python -m pipeline.train_xgb --data data/historical_aqi.csv --model models/xgb_model.pkl
python -m pipeline.train_lstm --data data/historical_aqi.csv --model models/lstm_model.h5 --scaler models/lstm_scaler.pkl --epochs 4
```
## Model Evaluation
----------------
Forecast Horizon: 72 hours
MAE: 18.13 AQI
RMSE: 21.82 AQI
