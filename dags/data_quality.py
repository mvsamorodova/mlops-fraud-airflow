"""DAG проверки размеченной витрины."""

import os
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from pipeline.quality.check_dataset import extract_quality_snapshot, check_dataset
from pipeline.quality.render_report import render_report, enforce_quality


@dag(
    dag_id="data_quality", schedule=None if os.getenv("DAG_SCHEDULE") == "manual" else os.getenv("DAG_SCHEDULE", "*/5 * * * *"),
    start_date=datetime(2024, 1, 1), catchup=False, max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=15)}, tags=["mlops", "dq"],
)
def data_quality():
    snapshot = task(extract_quality_snapshot)()
    checked = task(check_dataset)(snapshot)
    report = task(render_report)(checked)
    task(enforce_quality)(report)


data_quality()
