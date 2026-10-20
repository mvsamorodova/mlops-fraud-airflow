"""Ограничение глубины истории."""

from datetime import date
from dateutil.relativedelta import relativedelta
from pipeline.db import lock_state, read_training_state, transaction
from pipeline.settings import HISTORY_MONTHS, WINDOW_MONTHS


def prune_old_months(batch_info):
    """Удаляет строки до границы хранения; повтор DELETE безопасен, нужное окно защищено."""
    cutoff = date.fromisoformat(batch_info["batch_month"]) - relativedelta(months=HISTORY_MONTHS - 1)
    with transaction() as conn:
        lock_state(conn)
        needed = read_training_state(conn)["next_batch_month"] - relativedelta(months=WINDOW_MONTHS - 1)
        if cutoff > needed:
            raise ValueError("Очистка затронет необработанное окно")
        with conn.cursor() as cur:
            cur.execute("DELETE FROM labeled_transactions WHERE transaction_time < %s", (cutoff,))
            deleted = cur.rowcount
    return {**batch_info, "cutoff": cutoff.isoformat(), "deleted_rows": deleted}
