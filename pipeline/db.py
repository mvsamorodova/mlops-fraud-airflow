"""Витрина и журнал месяцев в PostgreSQL."""

import io
import os
from contextlib import contextmanager
from datetime import date

import pandas as pd
import psycopg2
from dateutil.relativedelta import relativedelta

from pipeline.settings import WINDOW_MONTHS

LABELED_COLUMNS = [
    "transaction_id", "transaction_time", "merch", "cat_id", "amount",
    "name_1", "name_2", "gender", "street", "one_city", "us_state", "post_code",
    "lat", "lon", "population_city", "jobs", "merchant_lat", "merchant_lon", "target",
]


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "fraud_db"),
        user=os.getenv("POSTGRES_USER", "fraud_user"),
        password=os.getenv("POSTGRES_PASSWORD", "fraud_pass"),
    )


@contextmanager
def transaction():
    conn = get_conn()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def lock_state(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(7001)")


def month_end_exclusive(month_start: date) -> date:
    return month_start + relativedelta(months=1)


def read_training_state(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT window_start, window_end, next_batch_month FROM training_state WHERE id = 1")
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("Сначала запустите data_loader")
    return dict(zip(["window_start", "window_end", "next_batch_month"], row))


def advance_training_state(conn, batch_month):
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE training_state SET window_start=%s, window_end=%s,
            next_batch_month=%s, updated_at=CURRENT_TIMESTAMP
            WHERE id=1 AND next_batch_month=%s""",
            (batch_month - relativedelta(months=WINDOW_MONTHS - 1), batch_month,
             month_end_exclusive(batch_month), batch_month),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Указатель месяца изменился: публикация отменена")


def copy_labeled_frame(conn, frame):
    buffer = io.StringIO()
    frame.loc[:, LABELED_COLUMNS].to_csv(buffer, index=False, header=False,
                                        date_format="%Y-%m-%d %H:%M:%S")
    buffer.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY labeled_transactions ({', '.join(LABELED_COLUMNS)}) FROM STDIN WITH (FORMAT CSV)",
            buffer,
        )


def read_frame(conn, start, end):
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(LABELED_COLUMNS)} FROM labeled_transactions "
            "WHERE transaction_time >= %s AND transaction_time < %s ORDER BY transaction_time, transaction_id",
            (start, end),
        )
        return pd.DataFrame(cur.fetchall(), columns=LABELED_COLUMNS)


def reserve_month(conn, kind, run_id, proposed_month):
    """Закрепляет месяц за run_id; незавершённый месяц подхватывается новым запуском."""
    with conn.cursor() as cur:
        cur.execute("SELECT batch_month FROM run_requests WHERE kind=%s AND run_id=%s", (kind, run_id))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute("SELECT batch_month FROM pipeline_runs WHERE kind=%s AND status='pending' ORDER BY batch_month LIMIT 1", (kind,))
        row = cur.fetchone()
        month = row[0] if row else proposed_month
        cur.execute("INSERT INTO pipeline_runs(kind, batch_month) VALUES (%s,%s) ON CONFLICT DO NOTHING", (kind, month))
        cur.execute("INSERT INTO run_requests(kind, run_id, batch_month) VALUES (%s,%s,%s)", (kind, run_id, month))
        return month


def run_status(conn, kind, month):
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM pipeline_runs WHERE kind=%s AND batch_month=%s", (kind, month))
        return cur.fetchone()[0]


def finish_run(conn, kind, month):
    with conn.cursor() as cur:
        cur.execute("UPDATE pipeline_runs SET status='complete' WHERE kind=%s AND batch_month=%s", (kind, month))
