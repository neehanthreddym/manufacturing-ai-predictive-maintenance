import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics(y, p: np.ndarray, threshold: float) -> dict:
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "false_negative_rate": float(fn / (fn + tp)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "brier_score": brier_score_loss(y, p),
    }


def save_figure(name: str) -> None:
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{name}.png", dpi=140)
    plt.close()


def final_evaluation(y, p: np.ndarray, threshold: float) -> dict:
    result = {
        "at_0_5": metrics(y, p, 0.5),
        "operational": metrics(y, p, threshold),
        "evaluation_note": (
            "Held-out test after validation-only selection; no refitting on validation/test."
        ),
    }
    Path("reports/final_metrics.json").write_text(json.dumps(result, indent=2))
    ConfusionMatrixDisplay.from_predictions(y, p >= threshold)
    save_figure("confusion_matrix")
    PrecisionRecallDisplay.from_predictions(y, p)
    save_figure("precision_recall_curve")
    RocCurveDisplay.from_predictions(y, p)
    save_figure("roc_curve")
    return result


def calibration_plot(y, raw: np.ndarray, calibrated: np.ndarray) -> None:
    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(y, raw, n_bins=8, name="Uncalibrated validation", ax=ax)
    CalibrationDisplay.from_predictions(
        y, calibrated, n_bins=8, name="CV sigmoid validation", ax=ax
    )
    save_figure("calibration_curve")
