"""Параметры витрины, обучения и артефактов."""

import os
from pathlib import Path

TRAINING_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/opt/airflow/training_data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/opt/airflow/models"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/opt/airflow/reports"))
FRAUD_SRC = os.getenv("FRAUD_SRC", "/opt/airflow/fraud_src")
CURRENT_DATASET = TRAINING_DIR / "current.parquet"
LABEL = "target"
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "10000"))
HISTORY_MONTHS = int(os.getenv("HISTORY_MONTHS", "36"))
WINDOW_MONTHS = int(os.getenv("WINDOW_MONTHS", "3"))
CAT_FEATURES = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs"]
if WINDOW_MONTHS < 2 or HISTORY_MONTHS < WINDOW_MONTHS or SAMPLE_SIZE < 1:
    raise ValueError("Нужны WINDOW_MONTHS >= 2, HISTORY_MONTHS >= WINDOW_MONTHS, SAMPLE_SIZE > 0")
