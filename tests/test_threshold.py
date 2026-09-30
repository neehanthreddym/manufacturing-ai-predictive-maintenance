import numpy as np
import pytest
from sklearn.metrics import fbeta_score

from predictmaint.config import Config
from predictmaint.data.validate import feature_matrix
from predictmaint.modeling.threshold import select_threshold
from predictmaint.modeling.train import split_data


def test_f2_optimum():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.3, 0.8])
    t, table = select_threshold(y, p)
    assert 0 <= t <= 1
    assert fbeta_score(y, p >= t, beta=2) == pytest.approx(table.f2.max())
    assert t == 0.3
    assert table.loc[table.threshold == 0.3, "false_negatives"].item() == 0


def test_split_isolation(raw):
    df = feature_matrix(raw)
    df["target"] = raw["Machine failure"]
    train, val, test = split_data(df, Config())
    assert not set(train.index) & set(val.index)
    assert not set(test.index) & (set(train.index) | set(val.index))
    assert len(train) + len(val) + len(test) == 60
    t, _ = select_threshold(val.target, np.linspace(0.01, 0.99, len(val)))
    test["target"] = 1 - test.target
    t2, _ = select_threshold(val.target, np.linspace(0.01, 0.99, len(val)))
    assert t == t2
