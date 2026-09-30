import numpy as np
import pytest

from predictmaint.config import Config
from predictmaint.data.preprocess import build_preprocessor, prepare_data
from predictmaint.data.validate import EXCLUDED, FEATURES, feature_matrix


def test_allowlist(raw):
    X = feature_matrix(raw)
    assert list(X) == FEATURES
    assert not set(EXCLUDED + ["Machine failure"]) & set(X.columns)
    altered = raw.copy()
    altered[EXCLUDED] = 999
    assert feature_matrix(altered).equals(X)


def test_encoding(raw):
    X = feature_matrix(raw)
    X.loc[0, "torque_nm"] = np.nan
    p = build_preprocessor().fit(X[X.machine_type != "H"])
    assert p.transform(X).shape == (60, 8)
    assert np.isfinite(p.transform(X)).all()
    X.loc[0, "machine_type"] = "Z"
    with pytest.raises(ValueError):
        p.transform(X)


def test_prepare(raw, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw.to_csv("raw.csv", index=False)
    config = Config()
    config.data.raw_path = "raw.csv"
    config.data.processed_path = "processed.csv"
    result = prepare_data(config)
    assert list(result) == FEATURES + ["target"]
    assert (tmp_path / "reports/data_validation.json").exists()
    config.data.raw_path = "absent.csv"
    with pytest.raises(FileNotFoundError):
        prepare_data(config)
