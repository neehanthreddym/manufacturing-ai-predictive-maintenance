import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from predictmaint.data.preprocess import build_preprocessor
from predictmaint.data.validate import feature_matrix
from predictmaint.modeling.artifacts import load_artifact, save_artifact
from predictmaint.modeling.evaluate import calibration_plot, final_evaluation, metrics
from predictmaint.modeling.explain import Explainer


def test_serialization(bundle, tmp_path):
    save_artifact(bundle, bundle["metadata"], tmp_path)
    loaded = load_artifact(tmp_path / "model.joblib")
    X = bundle["background"]
    assert np.allclose(loaded["pipeline"].predict_proba(X), bundle["pipeline"].predict_proba(X))
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "missing")
    joblib.dump({}, tmp_path / "bad")
    with pytest.raises(ValueError):
        load_artifact(tmp_path / "bad")
    bundle["threshold"] = 2
    joblib.dump(bundle, tmp_path / "bad")
    with pytest.raises(ValueError):
        load_artifact(tmp_path / "bad")


@pytest.mark.parametrize("model", [LogisticRegression(max_iter=500), DummyClassifier()])
def test_explanation_adapters(raw, model):
    X = feature_matrix(raw)
    pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)]).fit(
        X, raw["Machine failure"]
    )
    explainer = Explainer(pipe, X.iloc[:10])
    assert len(explainer.local(X.iloc[:1])) == 3
    assert explainer.values(X).shape == (60, 8)


def test_metrics_and_plots(tmp_path, monkeypatch, bundle):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.4, 0.9])
    result = final_evaluation(y, p, 0.4)
    assert result["operational"]["recall"] == 1
    assert metrics(y, p, 0.5)["false_negative_rate"] == 0.5
    calibration_plot(y, p, p)
    Explainer(bundle["base_pipeline"], bundle["background"]).global_plot(bundle["background"])
    assert len(list((tmp_path / "reports/figures").glob("*.png"))) == 5
