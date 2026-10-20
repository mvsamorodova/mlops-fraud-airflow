"""Обучение без использования проверочного месяца."""

import os
from catboost import CatBoostClassifier
from pipeline.settings import CAT_FEATURES, LABEL


def prepare_features(frame):
    return frame.drop(columns=[LABEL]), frame[LABEL]


def train_model(frame):
    x_train, y_train = prepare_features(frame)
    if y_train.nunique() != 2:
        raise ValueError("В train должны присутствовать оба класса")
    model = CatBoostClassifier(
        iterations=int(os.getenv("MODEL_ITERATIONS", "100")), depth=6,
        learning_rate=0.1, loss_function="Logloss", auto_class_weights="Balanced",
        random_seed=42, task_type="CPU", thread_count=2,
        allow_writing_files=False, verbose=False,
    )
    model.fit(x_train, y_train, cat_features=CAT_FEATURES)
    return model
