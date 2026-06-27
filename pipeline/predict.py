"""Live inference pipeline for AirSense."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pipeline.features import FEATURE_COLUMNS, future_feature_frame, latest_feature_window, risk_class_from_aqi, risk_label
from pipeline.fetch_data import build_historical_dataset, fetch_live_city_air_quality


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "historical_aqi.csv"
LSTM_PATH = ROOT / "models" / "lstm_model.h5"
LSTM_SCALER_PATH = ROOT / "models" / "lstm_scaler.pkl"
XGB_PATH = ROOT / "models" / "xgb_model.pkl"


def ensure_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return build_historical_dataset(DATA_PATH)
    return pd.read_csv(DATA_PATH)


def _statistical_forecast(window: np.ndarray, horizon: int = 72) -> np.ndarray:
    current = float(np.nanmean(window[-6:, 0] * 2.2))
    trend = float(np.nanmean(window[-12:, 0]) - np.nanmean(window[:12, 0]))
    hours = np.arange(1, horizon + 1)
    daily = 9 * np.sin((hours + window[-1, 7]) / 24 * 2 * np.pi)
    forecast = current + 0.12 * trend * (hours / horizon) + daily
    return np.clip(forecast, 5, 420)


def predict_lstm(window: np.ndarray) -> np.ndarray:
    try:
        from tensorflow.keras.models import load_model

        if LSTM_PATH.exists() and LSTM_SCALER_PATH.exists():
            scaler = joblib.load(LSTM_SCALER_PATH)
            model = load_model(LSTM_PATH, compile=False)
            scaled = scaler.transform(window.reshape(-1, len(FEATURE_COLUMNS))).reshape(1, *window.shape)
            return np.clip(model.predict(scaled, verbose=0)[0], 5, 500)
    except Exception:
        pass
    return _statistical_forecast(window)


def predict_risk_class(features: pd.DataFrame, current_aqi: float) -> int:
    # Future version:
    # if XGB_PATH.exists():
    #     model = joblib.load(XGB_PATH)
    #     pred = model.predict(features[FEATURE_COLUMNS].tail(1))[0]
    #     return int(pred)

    return risk_class_from_aqi(float(current_aqi))


def health_recommendations(risk_id: int, row: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    risk = risk_label(risk_id)
    if risk == "Good":
        recs.append("Air quality is favorable for normal outdoor activity.")
    elif risk == "Moderate":
        recs.append("Sensitive groups should keep prolonged outdoor exertion light.")
    elif risk == "Unhealthy":
        recs.append("Limit outdoor exposure and keep windows closed during peak pollution hours.")
    else:
        recs.append("Avoid outdoor activity, use filtered indoor air, and follow local health advisories.")

    if float(row.get("pm25", 0)) >= 35:
        recs.append("High PM2.5: wear an N95 mask outdoors and consider an indoor HEPA purifier.")
    if float(row.get("pm10", 0)) >= 150:
        recs.append("High PM10: reduce dust exposure and avoid busy roads where possible.")
    if float(row.get("o3", 0)) >= 70:
        recs.append("High ozone: avoid outdoor exercise between 10am and 4pm.")
    if float(row.get("no2", 0)) >= 100:
        recs.append("High NO2: avoid heavy traffic corridors and ventilate away from roadside air.")
    if float(row.get("wind_speed", 0)) < 1.0:
        recs.append("Low wind may allow pollutants to accumulate; check conditions before commuting.")
    return recs


def run_prediction(city: str, waqi_token: str = "", owm_api_key: str = "") -> dict[str, Any]:
    data = ensure_data()
    live = fetch_live_city_air_quality(city, waqi_token=waqi_token, owm_api_key=owm_api_key, allow_fallback=True)
    window = latest_feature_window(data, live["city"], live)
    forecast = predict_lstm(window)
    future = future_feature_frame(live, forecast)
    risk_id = predict_risk_class(future, float(live["aqi"]))
    return {
        "live": live,
        "forecast": future,
        "risk_id": risk_id,
        "risk_label": risk_label(risk_id),
        "recommendations": health_recommendations(risk_id, live),
    }

