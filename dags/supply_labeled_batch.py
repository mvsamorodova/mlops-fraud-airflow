"""DAG поставки нового размеченного месяца."""

import os
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from pipeline.supply.sample_next_month import sample_next_month
from pipeline.supply.append_labeled_batch import append_labeled_batch
from pipeline.supply.prune_old_months import prune_old_months


@dag(
    dag_id="supply_labeled_batch", schedule=None if os.getenv("DAG_SCHEDULE") == "manual" else os.getenv("DAG_SCHEDULE", "*/5 * * * *"),
    start_date=datetime(2024, 1, 1), catchup=False, max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
    default_args={"retries": 2, "retry_delay": timedelta(seconds=15)}, tags=["mlops", "etl"],
)
def supply_labeled_batch():
    sampled = task(sample_next_month)()
    appended = task(append_labeled_batch)(sampled)
    task(prune_old_months)(appended)


supply_labeled_batch()
