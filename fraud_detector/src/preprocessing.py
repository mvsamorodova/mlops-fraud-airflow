"""Общий препроцессинг обучения и онлайн-скоринга."""

import numpy as np
import pandas as pd

CAT_FEATURES = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs"]
NUM_FEATURES = ["amount", "population_city", "lat", "lon", "merchant_lat", "merchant_lon"]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def run_preproc(frame):
    """Строит stateless-признаки; target сохраняется только для обучения."""
    features = pd.DataFrame(index=frame.index)
    for column in CAT_FEATURES:
        features[column] = frame[column].fillna("unknown").astype(str)
    for column in NUM_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce")
        features[f"{column}_missing"] = values.isna().astype(int)
        features[column] = values.fillna(0)
    times = pd.to_datetime(frame["transaction_time"], errors="raise")
    if times.isna().any():
        raise ValueError("transaction_time содержит пропуски")
    features["amount_log"] = np.log1p(features["amount"].clip(lower=0))
    features["population_log"] = np.log1p(features["population_city"].clip(lower=0))
    hour = times.dt.hour + times.dt.minute / 60
    weekday = times.dt.dayofweek
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    features["weekend"] = (weekday >= 5).astype(int)
    features["night"] = (times.dt.hour < 6).astype(int)
    features["haversine"] = haversine(features["lat"], features["lon"],
                                       features["merchant_lat"], features["merchant_lon"])
    if "target" in frame:
        features["target"] = frame["target"].to_numpy()
    return features
