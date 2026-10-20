"""Обучение и сохранение кандидата."""

import os
import pandas as pd
from pipeline.model import train_model
from pipeline.settings import MODELS_DIR


def fit_catboost(preproc_info):
    """Читает только train, сохраняет кандидата месяца; повтор использует сохранённый бинарник."""
    version = "v" + preproc_info["batch_month"][:7].replace("-", "")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    candidate = MODELS_DIR / f"my_catboost_{version}.cbm"
    if not candidate.exists():
        model = train_model(pd.read_parquet(preproc_info["train_proc_path"]))
        temp = candidate.with_suffix(".tmp")
        model.save_model(str(temp))
        os.replace(temp, candidate)
    return {**preproc_info, "version": version, "model_path": str(candidate)}
