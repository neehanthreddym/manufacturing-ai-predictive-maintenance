"""Only load trusted local artifacts: joblib/pickle can execute code."""

import json
from pathlib import Path

import joblib


def save_artifact(bundle: dict, metadata: dict, directory: str | Path = "artifacts") -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path / "model.joblib")
    (path / "model_metadata.json").write_text(json.dumps(metadata, indent=2))


def load_artifact(path: str | Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError("Model artifact missing; run make train")
    bundle = joblib.load(path)
    required = {"pipeline", "base_pipeline", "threshold", "config", "metadata", "background"}
    if not isinstance(bundle, dict) or not required <= bundle.keys():
        raise ValueError("Invalid model artifact")
    if not 0 <= bundle["threshold"] <= 1:
        raise ValueError("Invalid artifact threshold")
    return bundle
