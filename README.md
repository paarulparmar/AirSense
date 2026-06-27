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

## API Keys

Create two free keys:

- WAQI: https://aqicn.org/api
- OpenWeatherMap: https://openweathermap.org/api

Use these secret names:

```toml
WAQI_TOKEN = "your_waqi_token"
OPENWEATHER_API_KEY = "your_openweathermap_key"
```

The app also accepts `AQICN_TOKEN` and `OWM_API_KEY` as aliases.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For local secrets, create `.streamlit/secrets.toml`:

```toml
WAQI_TOKEN = "your_waqi_token"
OPENWEATHER_API_KEY = "your_openweathermap_key"
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

If model files are missing or incompatible on Streamlit Cloud, AirSense uses robust statistical fallback inference so the public app still runs.

## Deploy to Streamlit Community Cloud

1. Create a public GitHub repository named `airsense`.
2. Push all files in this folder to the repository.
3. Go to https://share.streamlit.io.
4. Click **New app**.
5. Connect your GitHub repository.
6. Set the main file path to `app.py`.
7. Open **Advanced settings** and add secrets:

```toml
WAQI_TOKEN = "your_waqi_token"
OPENWEATHER_API_KEY = "your_openweathermap_key"
```

8. Click **Deploy**.
9. Streamlit will build the app and give you a permanent public URL like `https://airsense-parul.streamlit.app`.

## Notes

- API calls are cached for 30 minutes with `st.cache_data`.
- The live API keys are never hard-coded.
- The city fallback list includes all required demo cities, while OpenWeatherMap geocoding supports global city lookup when a key is configured.
- TensorFlow can take several minutes to install on a fresh Streamlit Cloud container. The app remains functional through fallback inference while trained model artifacts are being prepared.

