"""Извлечение следующего месяца без изменения указателя."""

import pandas as pd
from pipeline.artifacts import atomic_parquet
from pipeline.db import lock_state, month_end_exclusive, read_frame, read_training_state, reserve_month, transaction
from pipeline.settings import TRAINING_DIR


def extract_month_batch(run_id=None):
    """Читает витрину, сохраняет месячный snapshot; run_id закрепляет месяц при повторах."""
    if run_id is None:
        from airflow.operators.python import get_current_context
        run_id = get_current_context()["run_id"]
    with transaction() as conn:
        lock_state(conn)
        state = read_training_state(conn)
        month = reserve_month(conn, "train", run_id, state["next_batch_month"])
        path = TRAINING_DIR / "runs" / month.strftime("%Y%m") / "batch.parquet"
        if not path.exists():
            frame = read_frame(conn, month, month_end_exclusive(month))
            if frame.empty:
                from airflow.exceptions import AirflowSkipException
                raise AirflowSkipException(f"Месяц {month} ещё не поставлен")
            atomic_parquet(frame, path)
    return {"batch_path": str(path), "batch_month": month.isoformat(), "work_dir": str(path.parent)}
