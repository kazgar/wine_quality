import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_basic_metrics(
    estimator, X, y, dataset: pl.String, model_name: pl.String
) -> pl.DataFrame:
    y_pred = estimator.predict(X)
    y_true = y
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
    precision = precision_score(
        y_pred=y_pred, y_true=y_true, average="weighted", zero_division=np.nan
    )
    recall = recall_score(y_pred=y_pred, y_true=y_true, average="weighted", zero_division=np.nan)
    f1 = f1_score(y_pred=y_pred, y_true=y_true, average="weighted", zero_division=np.nan)
    balanced_accuracy = balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
    overall_test_set_performance = estimator.score(X, y)
    return pl.DataFrame(
        {
            "model": model_name,
            "dataset": dataset,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "overall_test_set_performance": overall_test_set_performance,
        }
    )
