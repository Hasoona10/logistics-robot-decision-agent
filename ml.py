"""
CPSC 481 — Logistics Robot ML module delay prediction only
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


FEATURE_COLUMNS = ["time_of_day", "zone_type", "congestion_level", "distance"]
TARGET_COLUMN = "delay_level"


def encode_dataset_csv(csv_path: str) -> tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """load CSV and replace every column with integer codes"""
    raw = pd.read_csv(csv_path)
    encoders: Dict[str, LabelEncoder] = {}
    encoded = raw.copy()
    for column in raw.columns:
        le = LabelEncoder()
        encoded[column] = le.fit_transform(raw[column])
        encoders[column] = le
    return encoded, encoders


class DelayPredictor:
    """Decision tree + encoders for delay_level use predict_delay() after train()."""

    def __init__(self, csv_path: Optional[str] = None):
        base = os.path.dirname(os.path.abspath(__file__))
        self._csv_path = csv_path or os.path.join(base, "data", "robot_delay_data.csv")
        self._model: Optional[DecisionTreeClassifier] = None
        self._encoders: Dict[str, LabelEncoder] = {}

    def train(self, random_state: int = 42) -> None:
        encoded, encoders = encode_dataset_csv(self._csv_path)
        X = encoded[FEATURE_COLUMNS]
        y = encoded[TARGET_COLUMN]
        model = DecisionTreeClassifier(random_state=random_state)
        model.fit(X.to_numpy(dtype=np.int64), y.to_numpy(dtype=np.int64))
        self._model = model
        self._encoders = encoders

    def predict_delay(
        self,
        time_of_day: str,
        zone_type: str,
        congestion_level: str,
        distance: str,
    ) -> str:
        """
        encode inputs with fitted LabelEncoders, predict, inverse-transform delay_level
        to "low", "medium", or "high".
        """
        if self._model is None:
            raise RuntimeError("Call train() before predict_delay.")

        enc = self._encoders

        def safe_transform(col: str, value: str) -> int:
            le = enc[col]
            if value not in list(le.classes_):
                raise ValueError(f"Unknown {col} label: {value!r}. Known: {list(le.classes_)}")
            return int(le.transform([value])[0])

        sample = [
            safe_transform("time_of_day", time_of_day),
            safe_transform("zone_type", zone_type),
            safe_transform("congestion_level", congestion_level),
            safe_transform("distance", distance),
        ]
        pred = int(self._model.predict(np.array([sample], dtype=np.int64))[0])
        label = enc[TARGET_COLUMN].inverse_transform([pred])[0]
        return str(label)

    def encoders(self) -> Dict[str, LabelEncoder]:
        return self._encoders


def load_and_train(csv_path: Optional[str] = None) -> DelayPredictor:
    p = DelayPredictor(csv_path)
    p.train()
    return p


if __name__ == "__main__":
    # Step 4 quick check: encoded frame (what the tree sees)
    _base = os.path.dirname(os.path.abspath(__file__))
    _csv = os.path.join(_base, "data", "robot_delay_data.csv")
    _encoded, _ = encode_dataset_csv(_csv)
    print("Encoded dataset (head) - Step 4:\n", _encoded.head(), sep="")

    _p = DelayPredictor(_csv)
    _p.train()
    print("\nExample predict_delay (afternoon, normal, low, medium):", _p.predict_delay("afternoon", "normal", "low", "medium"))
