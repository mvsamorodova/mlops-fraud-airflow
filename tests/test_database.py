import importlib
import json
import os
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from psycopg2 import sql

pytestmark = pytest.mark.skipif(os.getenv("TEST_POSTGRES") != "1", reason="Нужен PostgreSQL")


@pytest.fixture
def database(tmp_path, monkeypatch):
    from pipeline.db import get_conn
    name = "test_" + uuid4().hex[:12]
    admin = get_conn()
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    monkeypatch.setenv("POSTGRES_DB", name)
    monkeypatch.setenv("DATA_SOURCE", "demo")
    monkeypatch.setenv("MODEL_ITERATIONS", "10")
    modules = ["data_loader.app", "pipeline.retrain.extract_month_batch", "pipeline.retrain.fit_catboost",
               "pipeline.retrain.evaluate_model", "pipeline.retrain.publish_model", "pipeline.supply.sample_next_month"]
    for module_name in modules:
        module = importlib.import_module(module_name)
        for key, value in {"TRAINING_DIR": tmp_path / "training", "MODELS_DIR": tmp_path / "models",
                           "CURRENT_DATASET": tmp_path / "training/current.parquet"}.items():
            if hasattr(module, key):
                monkeypatch.setattr(module, key, value)
    from data_loader.app import main
    main()
    yield tmp_path
    with admin.cursor() as cur:
        cur.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
    admin.close()


def query(statement):
    from pipeline.db import transaction
    with transaction() as conn, conn.cursor() as cur:
        cur.execute(statement)
        return cur.fetchall()


def candidate(run_id):
    from pipeline.retrain.extract_month_batch import extract_month_batch
    from pipeline.retrain.slide_training_window import slide_training_window
    from pipeline.retrain.time_train_test_split import time_train_test_split
    from pipeline.retrain.preprocess_dataset import preprocess_dataset
    from pipeline.retrain.fit_catboost import fit_catboost
    from pipeline.retrain.evaluate_model import evaluate_model
    return evaluate_model(fit_catboost(preprocess_dataset(time_train_test_split(slide_training_window(extract_month_batch(run_id))))))


def test_supply_replay_and_pruning(database, monkeypatch):
    from pipeline.supply.sample_next_month import sample_next_month
    from pipeline.supply.append_labeled_batch import append_labeled_batch
    from pipeline.supply import prune_old_months as pruning
    before = query("SELECT count(*) FROM labeled_transactions")[0][0]
    batch = sample_next_month("supply-test")
    append_labeled_batch(batch)
    append_labeled_batch(batch)
    assert sample_next_month("supply-test") == batch
    assert query("SELECT count(*) FROM labeled_transactions")[0][0] == before + batch["rows"]
    assert query("SELECT count(*)-count(DISTINCT transaction_id) FROM labeled_transactions")[0][0] == 0
    monkeypatch.setattr(pruning, "HISTORY_MONTHS", 6)
    pruning.prune_old_months(batch)
    pruning.prune_old_months(batch)
    assert query("SELECT min(transaction_time)::date FROM labeled_transactions")[0][0] >= date(2023, 2, 1)


def test_publish_recovers_after_pointer_write(database, monkeypatch):
    from pipeline.retrain import publish_model as publisher
    info = candidate("train-crash")
    info["accepted"] = True
    before = query("SELECT next_batch_month FROM training_state")[0][0]
    advance = publisher.advance_training_state
    def fail(*args):
        raise RuntimeError("simulated crash after file publication")
    monkeypatch.setattr(publisher, "advance_training_state", fail)
    with pytest.raises(RuntimeError):
        publisher.publish_model(info)
    assert query("SELECT next_batch_month FROM training_state")[0][0] == before
    assert (database / "models/latest.txt").read_text().strip() == info["version"]
    monkeypatch.setattr(publisher, "advance_training_state", advance)
    publisher.publish_model(info)
    publisher.publish_model(info)
    assert query("SELECT next_batch_month FROM training_state")[0][0] == date(2023, 5, 1)
    from pipeline.retrain.extract_month_batch import extract_month_batch
    assert extract_month_batch("train-crash")["batch_month"] == "2023-04-01"
    assert extract_month_batch("train-next")["batch_month"] == "2023-05-01"


def test_rejected_candidate_keeps_production(database):
    from pipeline.retrain.publish_model import publish_model
    from pipeline.retrain.extract_month_batch import extract_month_batch
    from pipeline.retrain.slide_training_window import slide_training_window
    from pipeline.retrain.time_train_test_split import time_train_test_split
    from pipeline.retrain.preprocess_dataset import preprocess_dataset
    from pipeline.retrain.fit_catboost import fit_catboost
    from pipeline.retrain.evaluate_model import evaluate_model
    from pipeline.model import train_model
    processed = preprocess_dataset(time_train_test_split(slide_training_window(extract_month_batch("reject"))))
    corrupted = pd.read_parquet(processed["train_proc_path"])
    corrupted["target"] = 1 - corrupted["target"]
    trained = fit_catboost(processed)
    train_model(corrupted).save_model(trained["model_path"])
    info = evaluate_model(trained)
    assert not info["accepted"]
    assert info["candidate_metrics"]["average_precision"] < info["previous_metrics"]["average_precision"]
    before = (database / "models/latest.txt").read_text()
    publish_model(info)
    assert (database / "models/latest.txt").read_text() == before
    assert query("SELECT next_batch_month FROM training_state")[0][0] == date(2023, 5, 1)
    assert Path(info["model_path"]).exists()


def test_pending_month_is_recovered_by_new_run(database):
    from pipeline.retrain.extract_month_batch import extract_month_batch
    first = extract_month_batch("failed-run")
    second = extract_month_batch("recovery-run")
    assert first == second


def test_supply_insert_failure_rolls_back(database):
    from psycopg2.errors import UniqueViolation
    from pipeline.supply.sample_next_month import sample_next_month
    from pipeline.supply.append_labeled_batch import append_labeled_batch
    batch = sample_next_month("insert-crash")
    frame = pd.read_parquet(batch["batch_path"])
    before = query("SELECT count(*) FROM labeled_transactions")[0][0]
    pd.concat([frame, frame.head(1)]).to_parquet(batch["batch_path"], index=False)
    with pytest.raises(UniqueViolation):
        append_labeled_batch(batch)
    assert query("SELECT count(*) FROM labeled_transactions")[0][0] == before
    assert query("SELECT status FROM pipeline_runs WHERE kind='supply'")[0][0] == "pending"
    frame.to_parquet(batch["batch_path"], index=False)
    append_labeled_batch(batch)
    assert query("SELECT count(*) FROM labeled_transactions")[0][0] == before + len(frame)
