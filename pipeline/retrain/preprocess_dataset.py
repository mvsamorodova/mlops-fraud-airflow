"""Препроцессинг тем же модулем, что и в скоринге."""

from pathlib import Path
import pandas as pd
from fraud_detector.src.preprocessing import run_preproc
from pipeline.artifacts import atomic_parquet


def preprocess_dataset(split_info):
    """Читает train/test, пишет признаки; stateless-преобразования одинаковы при каждом повторе."""
    work = Path(split_info["work_dir"])
    paths = {}
    for part in ("train", "test"):
        path = work / f"{part}_proc.parquet"
        atomic_parquet(run_preproc(pd.read_parquet(split_info[f"{part}_raw_path"])), path)
        paths[f"{part}_proc_path"] = str(path)
    return {**split_info, **paths}
