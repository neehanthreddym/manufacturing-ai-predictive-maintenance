"""Real fitting on a synthetic fixture; its temporary metrics are never portfolio results."""

import json

import numpy as np

from predictmaint.cli import main
from predictmaint.data.validate import FEATURES
from predictmaint.modeling.artifacts import load_artifact
from predictmaint.service.predictor import Predictor
from predictmaint.service.schemas import Telemetry


def test_full_quick_pipeline(raw, tmp_path, monkeypatch, telemetry):
    monkeypatch.chdir(tmp_path)
    raw.to_csv("raw.csv", index=False)
    (tmp_path / "config.yaml").write_text("data: {raw_path: raw.csv}\nmodel: {cv_folds: 2}\n")
    (tmp_path / "README.md").write_text("<!-- RESULTS_START -->\n<!-- RESULTS_END -->")
    main(["train", "--quick", "--config", "config.yaml"])
    metadata = json.loads((tmp_path / "artifacts/model_metadata.json").read_text())
    assert metadata["dataset_rows"] == 60
    assert sum(s["rows"] for s in metadata["splits"].values()) == 60
    assert set(metadata["feature_names"]) == set(FEATURES)
    assert 0 <= metadata["decision_threshold"] <= 1
    assert metadata["quick"]
    assert len(json.loads((tmp_path / "reports/model_comparison.json").read_text())) == 4
    assert len(list((tmp_path / "reports/figures").glob("*.png"))) == 6
    bundle = load_artifact(tmp_path / "artifacts/model.joblib")
    result = Predictor(tmp_path / "artifacts/model.joblib").predict(Telemetry(**telemetry))
    assert result["decision_threshold"] == bundle["threshold"]
    assert np.isfinite(result["failure_probability"])
    assert "Operational" in (tmp_path / "README.md").read_text()
    assert (tmp_path / "reports/resume_bullets.md").exists()
