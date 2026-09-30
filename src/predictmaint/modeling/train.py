import hashlib
import importlib.metadata
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from predictmaint.config import Config
from predictmaint.data.preprocess import build_preprocessor, prepare_data
from predictmaint.data.validate import EXCLUDED, FEATURES
from predictmaint.modeling.artifacts import save_artifact
from predictmaint.modeling.evaluate import calibration_plot, final_evaluation, metrics, save_figure
from predictmaint.modeling.explain import Explainer
from predictmaint.modeling.reporting import publish_results
from predictmaint.modeling.threshold import select_threshold


def split_data(
    frame: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainval, test = train_test_split(
        frame,
        test_size=config.split.test_size,
        stratify=frame.target,
        random_state=config.random_seed,
    )
    train, val = train_test_split(
        trainval,
        test_size=config.split.validation_size / (1 - config.split.test_size),
        stratify=trainval.target,
        random_state=config.random_seed,
    )
    return train, val, test


def train(config: Config, quick: bool = False) -> dict:
    logging.info("Training started; quick=%s", quick)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    df = prepare_data(config)
    train_df, val, test = split_data(df, config)
    X, y = train_df[FEATURES], train_df.target
    cv = StratifiedKFold(config.model.cv_folds, shuffle=True, random_state=config.random_seed)
    if y.value_counts().min() < config.model.cv_folds:
        raise ValueError("Insufficient minority examples for stratified CV")
    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=config.random_seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150 if quick else 300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=1,
            random_state=config.random_seed,
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=1,
            random_state=config.random_seed,
        ),
    }
    pipelines = {}
    rows = []
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        if name == "xgboost":
            grid = {
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.1] if quick else [0.05, 0.1],
                "model__n_estimators": [100] if quick else [150, 300],
                "model__subsample": [0.9],
                "model__colsample_bytree": [0.9],
                "model__min_child_weight": [1] if quick else [1, 5],
                "model__reg_lambda": [1.0],
            }
            search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=1)
            search.fit(X, y)
            pipe = search.best_estimator_
            Path("reports/search_results.json").write_text(
                json.dumps(
                    {
                        "best_params": search.best_params_,
                        "training_cv_average_precision": search.best_score_,
                        "quick": quick,
                    },
                    indent=2,
                )
            )
        else:
            pipe.fit(X, y)
        pipelines[name] = pipe
        p = pipe.predict_proba(val[FEATURES])[:, 1]
        threshold, _ = select_threshold(val.target, p)
        row = {"model": name, **metrics(val.target, p, threshold)}
        rows.append(row)
        logging.info(
            "Validation %s AP=%.4f recall=%.4f", name, row["average_precision"], row["recall"]
        )
    pd.DataFrame(rows).to_csv("reports/model_comparison.csv", index=False)
    Path("reports/model_comparison.json").write_text(json.dumps(rows, indent=2))
    selected = max(rows, key=lambda r: (r["average_precision"], r["recall"], r["f2"]))
    base = pipelines[selected["model"]]
    calibrated = CalibratedClassifierCV(
        estimator=base, method="sigmoid", cv=cv, ensemble=False, n_jobs=1
    )
    calibrated.fit(X, y)
    raw = base.predict_proba(val[FEATURES])[:, 1]
    cal = calibrated.predict_proba(val[FEATURES])[:, 1]
    calibration_plot(val.target, raw, cal)
    use_calibration = brier_score_loss(val.target, cal) <= brier_score_loss(val.target, raw)
    final = calibrated if use_calibration else base
    pv = cal if use_calibration else raw
    threshold, table = select_threshold(val.target, pv)
    table.to_csv("reports/threshold_analysis.csv", index=False)
    table.plot(x="threshold", y=["precision", "recall", "f2"])
    plt.axvline(threshold, color="black", linestyle="--")
    save_figure("threshold_analysis")
    # Every choice is frozen before the only held-out test probability computation.
    test_p = final.predict_proba(test[FEATURES])[:, 1]
    final_metrics = final_evaluation(test.target, test_p, threshold)
    splits = {
        name: {
            "rows": len(part),
            "positive": int(part.target.sum()),
            "negative": int((part.target == 0).sum()),
        }
        for name, part in [("train", train_df), ("validation", val), ("test", test)]
    }
    metadata = {
        "version": "0.1.0",
        "model_name": selected["model"],
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURES,
        "excluded_columns": EXCLUDED,
        "random_seed": config.random_seed,
        "decision_threshold": threshold,
        "validation_metrics": metrics(val.target, pv, threshold),
        "final_test_metrics": final_metrics,
        "configuration": config.model_dump(),
        "dataset_rows": len(df),
        "positive_prevalence": float(df.target.mean()),
        "dataset_sha256": hashlib.sha256(Path(config.data.raw_path).read_bytes()).hexdigest(),
        "splits": splits,
        "selection_reason": (
            "Highest validation average precision; recall then F2 break "
            "ties. Threshold maximizes validation F2; precision then "
            "larger threshold break ties."
        ),
        "calibration": {
            "used": bool(use_calibration),
            "method": "training-only out-of-fold sigmoid; ensemble=False",
            "validation_brier_raw": brier_score_loss(val.target, raw),
            "validation_brier_calibrated": brier_score_loss(val.target, cal),
        },
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ["scikit-learn", "xgboost", "shap", "pandas", "numpy"]
        },
        "quick": quick,
        "explanation_scope": "Uncalibrated base model SHAP; not additive to calibrated probability",
    }
    background = X.sample(min(100, len(X)), random_state=config.random_seed)
    Explainer(base, background).global_plot(
        val[FEATURES].sample(min(200, len(val)), random_state=config.random_seed)
    )
    bundle = {
        "pipeline": final,
        "base_pipeline": base,
        "threshold": threshold,
        "config": config.model_dump(),
        "metadata": metadata,
        "background": background,
    }
    save_artifact(bundle, metadata)
    m = final_metrics["operational"]
    bullets = [
        "Built a Python manufacturing failure-detection POC comparing four models on AI4I; "
        f"selected {selected['model']} achieved held-out average precision "
        f"{m['average_precision']:.3f}.",
        "Implemented training-only calibration and validation-selected "
        f"F2 threshold {threshold:.4f}; held-out recall {m['recall']:.3f}, "
        f"precision {m['precision']:.3f}, F2 {m['f2']:.3f}.",
        (
            "Exposed persisted inference, SHAP attribution and impact-weighted "
            "fleet ranking through FastAPI with offline tests and Docker "
            "packaging."
        ),
    ]
    Path("reports/resume_bullets.md").write_text(chr(10).join("- " + b for b in bullets) + chr(10))
    publish_results(metadata)
    logging.info("Training complete: %s threshold=%.5f", selected["model"], threshold)
    return metadata
