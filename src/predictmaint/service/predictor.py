import logging
import os
import time
from pathlib import Path

import pandas as pd

from predictmaint.config import Config
from predictmaint.data.validate import FEATURES
from predictmaint.modeling.artifacts import load_artifact
from predictmaint.modeling.explain import Explainer
from predictmaint.service.operator_explanation import operator_explanation
from predictmaint.service.risk import risk_decision
from predictmaint.service.schemas import Telemetry


class Predictor:
    def __init__(self, path: str | Path = "artifacts/model.joblib") -> None:
        self.bundle = load_artifact(path)
        self.config = Config.model_validate(self.bundle["config"])
        self.explainer = Explainer(self.bundle["base_pipeline"], self.bundle["background"])
        logging.info("Model artifact loaded; model=%s", self.bundle["metadata"].get("model_name"))

    def info(self) -> dict:
        metadata = self.bundle["metadata"]
        return {
            key: metadata.get(key)
            for key in [
                "model_name",
                "version",
                "training_timestamp",
                "feature_names",
                "decision_threshold",
                "final_test_metrics",
                "explanation_scope",
            ]
        }

    def predict(self, telemetry: Telemetry, impact_weight: float = 1.0) -> dict:
        start = time.perf_counter()
        frame = pd.DataFrame([telemetry.model_dump()]).loc[:, FEATURES]
        p = float(self.bundle["pipeline"].predict_proba(frame)[0, 1])
        result = risk_decision(p, self.bundle["threshold"], self.config.risk_policy, impact_weight)
        result["top_drivers"] = self.explainer.local(frame)
        mode = os.getenv("EXPLANATION_MODE", self.config.explanation.mode)
        result["operator_explanation"] = operator_explanation(result, mode)
        result["inference_latency_ms"] = (time.perf_counter() - start) * 1000
        logging.info("Prediction latency_ms=%.3f", result["inference_latency_ms"])
        return result
