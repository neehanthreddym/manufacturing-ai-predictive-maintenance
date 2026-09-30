import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score


def threshold_table(y: np.ndarray | pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.unique(np.r_[0.0, probabilities, 1.0]):
        pred = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "f2": fbeta_score(y, pred, beta=2, zero_division=0),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    y_validation: np.ndarray | pd.Series, validation_probabilities: np.ndarray
) -> tuple[float, pd.DataFrame]:
    table = threshold_table(y_validation, validation_probabilities)
    best = table.sort_values(
        ["f2", "precision", "threshold"], ascending=[False, False, False]
    ).iloc[0]
    return float(best.threshold), table
