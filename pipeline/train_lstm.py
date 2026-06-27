"""Train the AirSense LSTM AQI forecaster."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.features import FEATURE_COLUMNS, make_lstm_sequences


def build_model(input_shape: tuple[int, int], horizon: int = 72):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=False),
            Dropout(0.15),
            Dense(64, activation="relu"),
            Dense(horizon),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mae", metrics=["mse"])
    return model


def train_lstm(
    data_path: str | Path = "data/historical_aqi.csv",
    model_path: str | Path = "models/lstm_model.h5",
    scaler_path: str | Path = "models/lstm_scaler.pkl",
    epochs: int = 4,
    batch_size: int = 64,
) -> None:
    df = pd.read_csv(data_path)
    x, y = make_lstm_sequences(df)
    if len(x) == 0:
        raise ValueError("Not enough rows to build LSTM sequences.")

    scaler = StandardScaler()
    flat = x.reshape(-1, len(FEATURE_COLUMNS))
    scaler.fit(flat)
    x_scaled = scaler.transform(flat).reshape(x.shape)

    model = build_model((x.shape[1], x.shape[2]), y.shape[1])
    model.fit(x_scaled, y, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=2)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AirSense LSTM model.")
    parser.add_argument("--data", default="data/historical_aqi.csv")
    parser.add_argument("--model", default="models/lstm_model.h5")
    parser.add_argument("--scaler", default="models/lstm_scaler.pkl")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    train_lstm(args.data, args.model, args.scaler, args.epochs)


if __name__ == "__main__":
    main()
