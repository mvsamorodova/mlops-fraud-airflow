"""Чтение CSV семинара и воспроизводимые демонстрационные данные."""

from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.db import LABELED_COLUMNS

ALIASES = {
    "trans_date_trans_time": "transaction_time", "merchant": "merch",
    "category": "cat_id", "amt": "amount", "first": "name_1", "last": "name_2",
    "city": "one_city", "state": "us_state", "zip": "post_code", "long": "lon",
    "city_pop": "population_city", "job": "jobs", "merch_lat": "merchant_lat",
    "merch_long": "merchant_lon", "is_fraud": "target", "trans_num": "transaction_id",
}


def demo_frame(months=6, rows_per_month=2000):
    rng = np.random.default_rng(42)
    frames = []
    for month in pd.date_range("2023-01-01", periods=months, freq="MS"):
        n = rows_per_month
        amount = rng.lognormal(3.5, 1.0, n).round(2)
        hour = rng.integers(0, 24, n)
        category = rng.choice(["grocery", "online", "travel"], n)
        risk = 0.01 + 0.35 * (amount > 100) + 0.25 * ((hour < 5) & (category == "online"))
        target = (rng.random(n) < risk).astype(int)
        target[:2] = [0, 1]
        frames.append(pd.DataFrame({
            "transaction_id": [f"demo-{month:%Y%m}-{i}" for i in range(n)],
            "transaction_time": month + pd.to_timedelta(rng.integers(0, 28, n), unit="D") + pd.to_timedelta(hour, unit="h"),
            "merch": rng.choice(["shop_1", "shop_2", "shop_3"], n), "cat_id": category,
            "amount": amount, "name_1": "demo", "name_2": "demo", "gender": rng.choice(["F", "M"], n),
            "street": "demo", "one_city": "demo_city", "us_state": "CA", "post_code": 90001,
            "lat": 34 + rng.normal(0, .1, n), "lon": -118 + rng.normal(0, .1, n),
            "population_city": rng.integers(1000, 1000000, n), "jobs": rng.choice(["job_1", "job_2"], n),
            "merchant_lat": 34 + rng.normal(0, .2, n), "merchant_lon": -118 + rng.normal(0, .2, n),
            "target": target,
        }))
    return pd.concat(frames, ignore_index=True)


def load_csv(path):
    frame = pd.read_csv(Path(path))
    frame = frame.rename(columns={k: v for k, v in ALIASES.items() if v not in frame.columns})
    if "transaction_id" not in frame:
        frame["transaction_id"] = [f"csv-{i}" for i in range(len(frame))]
    missing = set(LABELED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"В CSV нет колонок: {sorted(missing)}")
    frame = frame[LABELED_COLUMNS].copy()
    frame["transaction_time"] = pd.to_datetime(frame["transaction_time"], errors="raise")
    return frame
