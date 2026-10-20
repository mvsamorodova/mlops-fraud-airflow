"""Подготовка следующего месяца."""

import calendar
import hashlib
import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.artifacts import atomic_parquet
from pipeline.db import LABELED_COLUMNS, lock_state, read_training_state, reserve_month, transaction
from pipeline.settings import HISTORY_MONTHS, SAMPLE_SIZE, TRAINING_DIR, WINDOW_MONTHS

def next_month_start(max_time: pd.Timestamp) -> pd.Timestamp:
    """Возвращает первое число месяца, следующего за месяцем max_time."""
    return (pd.Timestamp(max_time).to_period("M") + 1).to_timestamp()


def shift_into_month(frame: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    """Переписывает transaction_time в заданный месяц, сохраняя день, час и минуту.

    Если в целевом месяце нет такого дня, день обрезается до последнего числа.
    Секунды обнуляются. Исходный frame не меняется.
    """
    shifted = frame.copy()
    times = pd.to_datetime(shifted["transaction_time"])
    year = int(month_start.year)
    month = int(month_start.month)
    last_day = calendar.monthrange(year, month)[1]
    shifted["transaction_time"] = pd.to_datetime(
        {
            "year": year,
            "month": month,
            "day": times.dt.day.clip(upper=last_day),
            "hour": times.dt.hour,
            "minute": times.dt.minute,
        }
    )
    return shifted



def sample_next_month(run_id=None):
    """Читает витрину, пишет батч закреплённого месяца; snapshot и run_id защищают повтор."""
    if run_id is None:
        from airflow.operators.python import get_current_context
        run_id = get_current_context()["run_id"]
    with transaction() as conn:
        lock_state(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT max(transaction_time) FROM labeled_transactions")
            latest = cur.fetchone()[0]
        month = reserve_month(conn, "supply", run_id, next_month_start(latest).date())
        path = TRAINING_DIR / "supply" / f"{month:%Y%m}.parquet"
        if not path.exists():
            cutoff = month - relativedelta(months=HISTORY_MONTHS - 1)
            needed = read_training_state(conn)["next_batch_month"] - relativedelta(months=WINDOW_MONTHS - 1)
            if cutoff > needed:
                from airflow.exceptions import AirflowSkipException
                raise AirflowSkipException("Поставка ждёт переобучение: нельзя удалить нужную историю")
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {', '.join(LABELED_COLUMNS)} FROM labeled_transactions "
                    "ORDER BY md5(transaction_id || %s) LIMIT %s", (month.isoformat(), SAMPLE_SIZE),
                )
                frame = pd.DataFrame(cur.fetchall(), columns=LABELED_COLUMNS)
            if frame.empty:
                raise ValueError("Размеченная витрина пуста")
            batch = shift_into_month(frame, pd.Timestamp(month))
            batch["transaction_id"] = [
                hashlib.sha256(f"{month}:{value}".encode()).hexdigest() for value in frame.transaction_id
            ]
            atomic_parquet(batch, path)
    return {"batch_path": str(path), "batch_month": month.isoformat(), "rows": len(pd.read_parquet(path))}
