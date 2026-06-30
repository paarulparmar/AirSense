import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pipeline.predict import predict_lstm
from pipeline.features import latest_feature_window

# Load dataset
df = pd.read_csv("data/historical_aqi.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

cities = df["city"].unique()

actual = []
predicted = []

for city in cities:
    city_df = df[df["city"] == city].sort_values("timestamp")

    # Skip if not enough data
    if len(city_df) < 144:
        continue

    # Use the last 72 hours as ground truth
    train = city_df.iloc[:-72]
    test = city_df.iloc[-72:]

    window = train.tail(72)

    features = window[
        ["pm25", "pm10", "no2", "o3", "temp", "humidity", "wind_speed", "hour_of_day", "month"]
    ].values

    forecast = predict_lstm(features)

    actual.extend(test["aqi"].values[:len(forecast)])
    predicted.extend(forecast[:len(test)])

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print("=" * 40)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print("=" * 40)
