"""Train the AirSense XGBoost health-risk classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from pipeline.features import xgb_training_matrix


def train_xgb(
    data_path: str | Path = "data/historical_aqi.csv",
    model_path: str | Path = "models/xgb_model.pkl",
) -> None:
    from xgboost import XGBClassifier

    df = pd.read_csv(data_path)
    x, y = xgb_training_matrix(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=2,
    )
    model.fit(x_train, y_train)
    print(classification_report(y_test, model.predict(x_test)))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AirSense XGBoost risk classifier.")
    parser.add_argument("--data", default="data/historical_aqi.csv")
    parser.add_argument("--model", default="models/xgb_model.pkl")
    args = parser.parse_args()
    train_xgb(args.data, args.model)


if __name__ == "__main__":
    main()

