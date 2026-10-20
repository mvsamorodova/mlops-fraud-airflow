"""Сравнение кандидата и production на одном отложенном месяце."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from pipeline.artifacts import atomic_json
from pipeline.model import prepare_features
from pipeline.settings import MODELS_DIR


def calculate_metrics(model, features, target):
    scores = model.predict_proba(features)[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("Модель вернула некорректные вероятности")
    predicted = scores > .5
    return {
        "average_precision": float(average_precision_score(target, scores)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
    }


def should_publish(candidate, previous):
    return candidate["average_precision"] >= previous["average_precision"]


def evaluate_model(train_info):
    """Читает кандидата, production и test; сохраняет решение один раз, повтор его не пересчитывает."""
    decision_path = Path(train_info["work_dir"]) / "evaluation.json"
    if decision_path.exists():
        return json.loads(decision_path.read_text())
    test = pd.read_parquet(train_info["test_proc_path"])
    features, target = prepare_features(test)
    if target.nunique() != 2:
        raise ValueError("Нужны оба класса в проверочном месяце")
    candidate = CatBoostClassifier().load_model(train_info["model_path"])
    active = (MODELS_DIR / "latest.txt").read_text().strip()
    metadata = json.loads((MODELS_DIR / f"my_catboost_{active}.metrics.json").read_text())
    if pd.Timestamp(metadata["train_end"]) >= pd.Timestamp(train_info["test_month"]):
        raise ValueError("Production обучалась на проверочном месяце: сравнение недопустимо")
    previous = CatBoostClassifier().load_model(str(MODELS_DIR / f"my_catboost_{active}.cbm"))
    candidate_metrics = calculate_metrics(candidate, features, target)
    previous_metrics = calculate_metrics(previous, features, target)
    result = {**train_info, "previous_version": active, "candidate_metrics": candidate_metrics,
              "previous_metrics": previous_metrics, "accepted": should_publish(candidate_metrics, previous_metrics),
              "train_rows": len(pd.read_parquet(train_info["train_proc_path"])), "test_rows": len(test)}
    atomic_json(decision_path, result)
    return result
