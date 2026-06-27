"""Feature engineering helpers for AirSense models."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["pm25", "pm10", "no2", "o3", "temp", "humidity", "wind_speed", "hour_of_day", "month"]
TARGET_COLUMN = "aqi"
RISK_LABELS = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]


def risk_class_from_aqi(aqi: float) -> int:
    if aqi <= 50:
        return 0
    elif aqi <= 100:
        return 1
    elif aqi <= 150:
        return 2
    elif aqi <= 200:
        return 3
    elif aqi <= 300:
        return 4
    else:
        return 5


def risk_label(class_id: int) -> str:
    return RISK_LABELS[int(np.clip(class_id, 0, len(RISK_LABELS) - 1))]


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values(["city", "timestamp"]).reset_index(drop=True)
    out["hour_of_day"] = out["timestamp"].dt.hour
    out["month"] = out["timestamp"].dt.month
    out["risk_class"] = out[TARGET_COLUMN].apply(risk_class_from_aqi)
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[FEATURE_COLUMNS + [TARGET_COLUMN]] = out[FEATURE_COLUMNS + [TARGET_COLUMN]].interpolate().ffill().bfill()
    return out


def make_lstm_sequences(
    df: pd.DataFrame,
    lookback: int = 72,
    horizon: int = 72,
    max_sequences_per_city: int | None = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """Build LSTM windows: last 72 hours of features -> next 72 AQI values."""

    prepared = prepare_frame(df)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _, group in prepared.groupby("city", sort=False):
        group = group.sort_values("timestamp")
        values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        targets = group[TARGET_COLUMN].to_numpy(dtype=np.float32)
        max_start = len(group) - lookback - horizon
        if max_start <= 0:
            continue
        starts = np.arange(max_start)
        if max_sequences_per_city and len(starts) > max_sequences_per_city:
            starts = np.linspace(0, max_start - 1, max_sequences_per_city, dtype=int)
        for start in starts:
            xs.append(values[start : start + lookback])
            ys.append(targets[start + lookback : start + lookback + horizon])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def xgb_training_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    prepared = prepare_frame(df)
    return prepared[FEATURE_COLUMNS], prepared["risk_class"]


def latest_feature_window(df: pd.DataFrame, city: str, live_row: dict, lookback: int = 72) -> np.ndarray:
    prepared = prepare_frame(df)
    city_rows = prepared[prepared["city"].str.lower() == city.lower()].tail(max(lookback - 1, 0))
    live = pd.DataFrame([live_row])
    live["timestamp"] = pd.Timestamp.utcnow()
    live["city"] = city
    live["aqi"] = live.get("aqi", np.nan)
    merged = pd.concat([city_rows, prepare_frame(live)], ignore_index=True)
    if len(merged) < lookback:
        pad = pd.concat([merged.iloc[[0]]] * (lookback - len(merged)), ignore_index=True)
        merged = pd.concat([pad, merged], ignore_index=True)
    return merged.tail(lookback)[FEATURE_COLUMNS].to_numpy(dtype=np.float32)


def future_feature_frame(last_row: dict, forecast: np.ndarray) -> pd.DataFrame:
    now = pd.Timestamp.utcnow().floor("h")
    rows = []
    for i, aqi in enumerate(forecast, start=1):
        ts = now + pd.Timedelta(hours=i)
        rows.append(
            {
                "timestamp": ts,
                "aqi": float(aqi),
                "pm25": float(last_row["pm25"]) * (float(aqi) / max(float(last_row["aqi"]), 1.0)),
                "pm10": float(last_row["pm10"]) * (float(aqi) / max(float(last_row["aqi"]), 1.0)),
                "no2": float(last_row["no2"]),
                "o3": float(last_row["o3"]),
                "temp": float(last_row["temp"]),
                "humidity": float(last_row["humidity"]),
                "wind_speed": float(last_row["wind_speed"]),
                "hour_of_day": ts.hour,
                "month": ts.month,
            }
        )
    return pd.DataFrame(rows)

