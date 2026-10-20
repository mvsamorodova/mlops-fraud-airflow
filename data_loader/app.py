"""Начальная загрузка витрины и обучение стартовой модели."""

import logging
import os
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from fraud_detector.src.preprocessing import run_preproc
from pipeline.artifacts import atomic_json, atomic_parquet, atomic_text
from pipeline.data import demo_frame, load_csv
from pipeline.db import copy_labeled_frame, lock_state, read_frame, read_training_state, transaction
from pipeline.model import train_model
from pipeline.settings import CURRENT_DATASET, MODELS_DIR, TRAINING_DIR, WINDOW_MONTHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_data():
    """Читает CSV или demo, пишет витрину и стартовое окно; повтор не перезаливает БД."""
    with transaction() as conn:
        lock_state(conn)
        with conn.cursor() as cur:
            cur.execute(Path("pipeline/schema.sql").read_text())
            cur.execute("SELECT count(*) FROM training_state")
            initialized = cur.fetchone()[0]
        if not initialized:
            source = os.getenv("DATA_SOURCE", "demo")
            if source not in {"demo", "csv"}:
                raise ValueError("DATA_SOURCE должен быть demo или csv")
            frame = demo_frame() if source == "demo" else load_csv(os.environ["LABELED_CSV_PATH"])
            from pipeline.quality.check_dataset import check_frame
            checks = check_frame(frame)
            failed = [row["rule"] for row in checks if not row["passed"]]
            if failed:
                raise ValueError(f"Исходные данные не прошли DQ: {failed}")
            periods = sorted(frame.transaction_time.dt.to_period("M").unique())
            if len(periods) < WINDOW_MONTHS or periods != list(pd.period_range(periods[0], periods[-1], freq="M")):
                raise ValueError("Нужны последовательные месяцы, не меньше WINDOW_MONTHS")
            start = periods[0].start_time.date()
            end = start + relativedelta(months=WINDOW_MONTHS - 1)
            copy_labeled_frame(conn, frame)
            with conn.cursor() as cur:
                cur.execute("INSERT INTO training_state VALUES (1,%s,%s,%s,CURRENT_TIMESTAMP)",
                            (start, end, end + relativedelta(months=1)))
            logger.info("Загружено %s строк, источник=%s", len(frame), source)
        state = read_training_state(conn)
        frame = read_frame(conn, state["window_start"], state["window_end"] + relativedelta(months=1))
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    if not CURRENT_DATASET.exists():
        atomic_parquet(frame, CURRENT_DATASET)
    stream_path = TRAINING_DIR / "stream.parquet"
    if not stream_path.exists():
        atomic_parquet(frame.sample(n=min(5000, len(frame)), random_state=42).drop(columns=["target"]), stream_path)
    return frame, state


def initialize_model(frame, state):
    """Обучает initial на ранних месяцах; существующий latest.txt не заменяет."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if (MODELS_DIR / "latest.txt").exists():
        return
    train = frame.loc[frame.transaction_time < pd.Timestamp(state["window_end"])]
    model = train_model(run_preproc(train))
    temp = MODELS_DIR / ".initial.cbm"
    model.save_model(str(temp))
    os.replace(temp, MODELS_DIR / "my_catboost_initial.cbm")
    atomic_json(MODELS_DIR / "my_catboost_initial.metrics.json", {
        "version": "initial", "train_end": str(train.transaction_time.max()), "source": os.getenv("DATA_SOURCE", "demo"),
    })
    atomic_text(MODELS_DIR / "latest.txt", "initial\n")
    logger.info("Стартовая модель сохранена")


def main():
    frame, state = initialize_data()
    initialize_model(frame, state)


if __name__ == "__main__":
    main()
