"""Временной train/test split."""

from pathlib import Path
import pandas as pd
from pipeline.artifacts import atomic_parquet


def time_train_test_split(window_info):
    """Читает окно, пишет ранние месяцы в train и последний в test; повтор детерминирован."""
    frame = pd.read_parquet(window_info["dataset_path"])
    periods = frame.transaction_time.dt.to_period("M")
    latest = pd.Period(window_info["batch_month"], freq="M")
    train = frame.loc[periods < latest]
    test = frame.loc[periods == latest]
    if train.empty or test.empty or train.transaction_time.max() >= test.transaction_time.min():
        raise ValueError("Некорректное временное разбиение")
    if train.target.nunique() != 2 or test.target.nunique() != 2:
        raise ValueError("В train и test нужны оба класса")
    work = Path(window_info["work_dir"])
    train_path, test_path = work / "train_raw.parquet", work / "test_raw.parquet"
    atomic_parquet(train, train_path)
    atomic_parquet(test, test_path)
    return {**window_info, "train_raw_path": str(train_path), "test_raw_path": str(test_path),
            "train_months": sorted(periods.loc[periods < latest].astype(str).unique()),
            "test_month": str(latest), "train_end": str(train.transaction_time.max())}
