"""Проверки схемы, значений и покрытия месяцев."""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.artifacts import atomic_json, atomic_parquet
from pipeline.db import LABELED_COLUMNS, read_frame, transaction
from pipeline.settings import REPORTS_DIR, TRAINING_DIR, WINDOW_MONTHS


def check_frame(frame):
    checks = []
    def add(rule, failed, details=""):
        checks.append({"rule": rule, "passed": failed == 0, "violations": int(failed), "details": details})
    missing = set(LABELED_COLUMNS) - set(frame.columns)
    add("schema", len(missing), ", ".join(sorted(missing)))
    if missing:
        return checks
    add("not_empty", int(frame.empty))
    add("required_not_null", frame[["transaction_id", "transaction_time", "amount", "target"]].isna().any(axis=1).sum())
    add("unique_transaction_id", frame.transaction_id.duplicated().sum())
    amounts = pd.to_numeric(frame.amount, errors="coerce")
    add("finite_nonnegative_amount", (~np.isfinite(amounts) | (amounts < 0)).sum())
    add("binary_target", (~frame.target.isin([0, 1])).sum())
    times = pd.to_datetime(frame.transaction_time, errors="coerce")
    add("valid_time", times.isna().sum())
    coords = frame[["lat", "lon", "merchant_lat", "merchant_lon"]].apply(pd.to_numeric, errors="coerce")
    limits = pd.Series({"lat": 90, "lon": 180, "merchant_lat": 90, "merchant_lon": 180})
    add("valid_coordinates", (~np.isfinite(coords) | (coords.abs() > limits)).any(axis=1).sum())
    periods = times.dropna().dt.to_period("M")
    expected = list(pd.period_range(periods.min(), periods.max(), freq="M")) if len(periods) else []
    add("consecutive_months", int(len(expected) < WINDOW_MONTHS or sorted(periods.unique()) != expected))
    counts = frame.assign(month=times.dt.to_period("M")).groupby("month").agg(rows=("target", "size"), classes=("target", "nunique"))
    add("monthly_rows_and_classes", int(counts.empty) + ((counts.rows < 100) | (counts.classes != 2)).sum(),
        "Не менее 100 строк и оба класса в каждом месяце")
    return checks


def extract_quality_snapshot(run_id=None):
    """Читает витрину в snapshot; повтор того же запуска использует сохранённый файл."""
    if run_id is None:
        from airflow.operators.python import get_current_context
        run_id = get_current_context()["run_id"]
    key = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    path = TRAINING_DIR / "quality" / f"{key}.parquet"
    if not path.exists():
        with transaction() as conn:
            conn.set_session(isolation_level="REPEATABLE READ")
            with conn.cursor() as cur:
                cur.execute("SELECT min(transaction_time), max(transaction_time) FROM labeled_transactions")
                start, end = cur.fetchone()
            if start is None:
                frame = pd.DataFrame(columns=LABELED_COLUMNS)
            else:
                frame = read_frame(conn, start, end + relativedelta(seconds=1))
            atomic_parquet(frame, path)
    return {"snapshot": str(path), "key": key, "run_id": run_id}


def check_dataset(info):
    """Читает snapshot, сохраняет результаты правил; повтор пересчитывает тот же срез."""
    frame = pd.read_parquet(info["snapshot"])
    result = {**info, "rows": len(frame), "checks": check_frame(frame)}
    path = REPORTS_DIR / "dq" / f"{info['key']}.json"
    atomic_json(path, result)
    return result
