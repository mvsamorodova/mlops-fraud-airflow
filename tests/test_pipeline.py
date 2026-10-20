from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detector.src.preprocessing import run_preproc
from pipeline.data import demo_frame, load_csv
from pipeline.model import train_model
from pipeline.quality.check_dataset import check_frame
from pipeline.retrain.evaluate_model import should_publish
from pipeline.retrain.time_train_test_split import time_train_test_split
from pipeline.supply.sample_next_month import shift_into_month


@pytest.fixture
def frame():
    return demo_frame(months=3, rows_per_month=150)


def test_shared_features_without_target_leakage(frame):
    frame.loc[0, "merch"] = None
    training = run_preproc(frame)
    online = run_preproc(frame.drop(columns=["target"]))
    pd.testing.assert_frame_equal(training.drop(columns=["target"]), online)
    assert "transaction_id" not in online and "name_1" not in online
    assert online.loc[0, "merch"] == "unknown"
    assert np.isfinite(online.select_dtypes(include="number")).all().all()
    assert {"amount_log", "hour_sin", "hour_cos", "night"} <= set(online)


@pytest.mark.parametrize("month,day", [("2024-02-01", 29), ("2023-02-01", 28), ("2024-01-01", 31)])
def test_calendar_month_boundary(month, day):
    source = pd.DataFrame({"transaction_time": pd.to_datetime(["2023-12-31 23:59:00"])})
    shifted = shift_into_month(source, pd.Timestamp(month))
    assert shifted.transaction_time.iloc[0].day == day
    assert shifted.transaction_time.iloc[0].hour == 23
    assert source.transaction_time.iloc[0].month == 12


def test_time_split_uses_last_month(frame, tmp_path):
    path = tmp_path / "window.parquet"
    frame.to_parquet(path)
    info = time_train_test_split({"dataset_path": str(path), "work_dir": str(tmp_path), "batch_month": "2023-03-01"})
    train, test = pd.read_parquet(info["train_raw_path"]), pd.read_parquet(info["test_raw_path"])
    assert train.transaction_time.max() < test.transaction_time.min()
    assert set(train.transaction_time.dt.month) == {1, 2}
    assert set(test.transaction_time.dt.month) == {3}


def test_quality_rules_and_failed_html(frame, tmp_path, monkeypatch):
    from pipeline.quality import render_report as module
    assert all(row["passed"] for row in check_frame(frame))
    frame.loc[0, "amount"] = -1
    frame.loc[1, "target"] = 3
    frame.loc[2, "lat"] = 100
    frame.loc[3, "transaction_id"] = frame.loc[4, "transaction_id"]
    frame.loc[5, "transaction_time"] = pd.NaT
    checks = check_frame(frame)
    failed = {row["rule"] for row in checks if not row["passed"]}
    assert {"finite_nonnegative_amount", "binary_target", "valid_coordinates", "unique_transaction_id", "valid_time"} <= failed
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path)
    report = module.render_report({"key": "bad", "run_id": "<unsafe>", "rows": len(frame), "checks": checks})
    with pytest.raises(ValueError):
        module.enforce_quality(report)
    html = Path(report["path"]).read_text()
    assert "FAIL" in html and "&lt;unsafe&gt;" in html


@pytest.mark.parametrize("new,old,expected", [(.5,.6,False), (.6,.6,True), (.7,.6,True)])
def test_quality_gate(new, old, expected):
    assert should_publish({"average_precision": new}, {"average_precision": old}) is expected


def test_scorer_hot_reload_and_failed_reload(frame, tmp_path, monkeypatch):
    from fraud_detector.src import scorer
    monkeypatch.setattr(scorer, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(scorer, "_model", None)
    monkeypatch.setattr(scorer, "_version", None)
    monkeypatch.setenv("MODEL_ITERATIONS", "10")
    model = train_model(run_preproc(frame))
    for version in ("first", "second"):
        model.save_model(str(tmp_path / f"my_catboost_{version}.cbm"))
    (tmp_path / "latest.txt").write_text("first")
    _, _, first = scorer.make_pred(run_preproc(frame.head(1)))
    (tmp_path / "latest.txt").write_text("second")
    _, _, second = scorer.make_pred(run_preproc(frame.head(1)))
    assert (first, second) == ("first", "second")
    (tmp_path / "latest.txt").write_text("broken")
    _, _, fallback = scorer.make_pred(run_preproc(frame.head(1)))
    assert fallback == "second"


def test_csv_aliases(frame, tmp_path):
    path = tmp_path / "input.csv"
    frame.rename(columns={"amount": "amt", "target": "is_fraud"}).to_csv(path, index=False)
    loaded = load_csv(path)
    assert len(loaded) == len(frame)
    assert {"amount", "target"} <= set(loaded)
