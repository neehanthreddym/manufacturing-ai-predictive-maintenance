"""SHAP explains base estimator output, not calibrated probabilities or causality."""

import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from predictmaint.modeling.evaluate import save_figure


class Explainer:
    def __init__(self, pipeline: Pipeline, background: pd.DataFrame) -> None:
        self.preprocessor = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.names = list(self.preprocessor.get_feature_names_out())
        bg = self.preprocessor.transform(background)
        if isinstance(self.model, LogisticRegression):
            self.explainer = shap.LinearExplainer(self.model, bg)
            self.units = "base_model_log_odds"
        elif isinstance(self.model, DummyClassifier):
            self.explainer = None
            self.units = "constant_baseline"
        else:
            self.explainer = shap.TreeExplainer(self.model)
            self.units = (
                "base_model_log_odds"
                if isinstance(self.model, XGBClassifier)
                else "base_model_probability"
            )

    def values(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(frame)
        if self.explainer is None:
            return np.zeros_like(transformed)
        vals = np.asarray(self.explainer.shap_values(transformed))
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        return vals

    def local(self, frame: pd.DataFrame) -> list[dict]:
        values = self.values(frame)[0]
        grouped = {}
        for name, value in zip(self.names, values):
            original = "machine_type" if name.startswith("machine_type_") else name
            grouped[original] = grouped.get(original, 0.0) + float(value)
        return [
            {
                "feature": name,
                "value": frame.iloc[0][name].item()
                if hasattr(frame.iloc[0][name], "item")
                else frame.iloc[0][name],
                "contribution": value,
                "units": self.units,
            }
            for name, value in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[
                :3
            ]
        ]

    def global_plot(self, frame: pd.DataFrame) -> None:
        transformed = pd.DataFrame(self.preprocessor.transform(frame), columns=self.names)
        shap.summary_plot(
            self.values(frame), transformed, show=False, rng=np.random.default_rng(42)
        )
        save_figure("shap_summary")
