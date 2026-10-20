"""Публикация принятой модели и фиксация обработанного месяца."""

import json
from datetime import date
from pathlib import Path
import pandas as pd

from pipeline.artifacts import atomic_json, atomic_parquet, atomic_text
from pipeline.db import advance_training_state, finish_run, lock_state, read_training_state, run_status, transaction
from pipeline.settings import CURRENT_DATASET, MODELS_DIR


def publish_model(train_info):
    """Пишет метрики и latest, затем состояние БД; повтор завершённого месяца ничего не меняет."""
    month = date.fromisoformat(train_info["batch_month"])
    version = train_info["version"]
    if not Path(train_info["model_path"]).is_file():
        raise FileNotFoundError("Кандидат не сохранён")
    with transaction() as conn:
        lock_state(conn)
        if run_status(conn, "train", month) == "complete":
            return version
        if read_training_state(conn)["next_batch_month"] != month:
            raise RuntimeError("Публикация устаревшего окна запрещена")
        active = (MODELS_DIR / "latest.txt").read_text().strip()
        if active not in {train_info["previous_version"], version}:
            raise RuntimeError("Production изменилась после оценки")
        atomic_json(MODELS_DIR / f"my_catboost_{version}.metrics.json", train_info)
        if train_info["accepted"]:
            atomic_text(MODELS_DIR / "latest.txt", version + "\n")
        atomic_parquet(pd.read_parquet(train_info["dataset_path"]), CURRENT_DATASET)
        advance_training_state(conn, month)
        finish_run(conn, "train", month)
    return version
