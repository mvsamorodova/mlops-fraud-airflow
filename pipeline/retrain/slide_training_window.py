"""Материализация скользящего окна."""

from datetime import date
from pathlib import Path
import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.artifacts import atomic_parquet
from pipeline.db import read_frame, transaction
from pipeline.settings import WINDOW_MONTHS


def slide_training_window(batch_info):
    """Читает историю и snapshot месяца; пишет фиксированное окно, повтор использует тот же файл."""
    path = Path(batch_info["work_dir"]) / "window.parquet"
    if not path.exists():
        month = date.fromisoformat(batch_info["batch_month"])
        start = month - relativedelta(months=WINDOW_MONTHS - 1)
        with transaction() as conn:
            history = read_frame(conn, start, month)
        frame = pd.concat([history, pd.read_parquet(batch_info["batch_path"])], ignore_index=True)
        frame = frame.sort_values(["transaction_time", "transaction_id"])
        periods = sorted(frame.transaction_time.dt.to_period("M").unique())
        expected = list(pd.period_range(start, month, freq="M"))
        if periods != expected:
            raise ValueError(f"Неполное окно: {periods}, ожидалось {expected}")
        atomic_parquet(frame, path)
    return {**batch_info, "dataset_path": str(path)}
