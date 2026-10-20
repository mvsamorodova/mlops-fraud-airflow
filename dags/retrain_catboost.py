"""DAG скользящего переобучения CatBoost."""

import os
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from pipeline.retrain.extract_month_batch import extract_month_batch
from pipeline.retrain.slide_training_window import slide_training_window
from pipeline.retrain.time_train_test_split import time_train_test_split
from pipeline.retrain.preprocess_dataset import preprocess_dataset
from pipeline.retrain.fit_catboost import fit_catboost
from pipeline.retrain.evaluate_model import evaluate_model
from pipeline.retrain.publish_model import publish_model


@dag(
    dag_id="retrain_catboost", schedule=None if os.getenv("DAG_SCHEDULE") == "manual" else os.getenv("DAG_SCHEDULE", "*/5 * * * *"),
    start_date=datetime(2024, 1, 1), catchup=False, max_active_runs=1,
    dagrun_timeout=timedelta(minutes=20),
    default_args={"retries": 2, "retry_delay": timedelta(seconds=15)}, tags=["mlops", "catboost"],
)
def retrain_catboost():
    extracted = task(extract_month_batch)()
    window = task(slide_training_window)(extracted)
    split = task(time_train_test_split)(window)
    processed = task(preprocess_dataset)(split)
    trained = task(fit_catboost)(processed)
    evaluated = task(evaluate_model)(trained)
    task(publish_model)(evaluated)


retrain_catboost()
