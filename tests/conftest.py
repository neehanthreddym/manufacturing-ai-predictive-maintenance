from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from predictmaint.config import Config
from predictmaint.data.preprocess import build_preprocessor
from predictmaint.data.validate import FEATURES, feature_matrix, validate_data
from predictmaint.modeling.artifacts import save_artifact
from predictmaint.service.app import create_app
from predictmaint.service.predictor import Predictor


@pytest.fixture(autouse=True)
def offline_explanations(monkeypatch):
    """Keep API tests offline even when local .env enables a live provider."""
    monkeypatch.setenv("EXPLANATION_MODE", "template")


@pytest.fixture
def raw():
    return pd.read_csv(Path(__file__).parent / "fixtures/sample_ai4i.csv")


@pytest.fixture
def telemetry():
    return {
        "machine_type": "M",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 125,
    }


@pytest.fixture
def bundle(raw):
    df, _ = validate_data(raw)
    X = feature_matrix(df)
    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    ).fit(X, df["Machine failure"])
    metadata = {
        "model_name": "fixture_random_forest",
        "feature_names": FEATURES,
        "decision_threshold": 0.3,
        "version": "test",
    }
    return {
        "pipeline": pipe,
        "base_pipeline": pipe,
        "threshold": 0.3,
        "config": Config().model_dump(),
        "background": X.iloc[:20],
        "metadata": metadata,
    }


@pytest.fixture
def predictor(bundle, tmp_path):
    save_artifact(bundle, bundle["metadata"], tmp_path)
    return Predictor(tmp_path / "model.joblib")


@pytest.fixture
def client(predictor):
    with TestClient(create_app(predictor)) as client:
        yield client
