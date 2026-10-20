"""Идемпотентная запись месячного батча."""

from datetime import date
import pandas as pd
from pipeline.db import copy_labeled_frame, finish_run, lock_state, run_status, transaction


def append_labeled_batch(batch_info):
    """Читает snapshot и пишет строки с журналом в одной транзакции; повтор не вставляет дубли."""
    month = date.fromisoformat(batch_info["batch_month"])
    with transaction() as conn:
        lock_state(conn)
        if run_status(conn, "supply", month) != "complete":
            copy_labeled_frame(conn, pd.read_parquet(batch_info["batch_path"]))
            finish_run(conn, "supply", month)
    return batch_info
