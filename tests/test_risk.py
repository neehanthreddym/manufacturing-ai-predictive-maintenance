import pytest

from predictmaint.config import RiskConfig
from predictmaint.service.risk import risk_decision


@pytest.mark.parametrize(
    "p,band",
    [
        (0, "LOW"),
        (0.24999, "LOW"),
        (0.25, "MEDIUM"),
        (0.5, "HIGH"),
        (0.75, "CRITICAL"),
        (1, "CRITICAL"),
    ],
)
def test_boundaries(p, band):
    result = risk_decision(p, 0.3, RiskConfig(), 2)
    assert result["risk_band"] == band and result["risk_score"] == 2 * p


@pytest.mark.parametrize(
    "p,w", [(-0.1, 1), (1.1, 1), (0.5, -1), (0.5, float("inf")), (float("nan"), 1)]
)
def test_invalid(p, w):
    with pytest.raises(ValueError):
        risk_decision(p, 0.3, RiskConfig(), w)


def test_priority():
    assert risk_decision(0.2, 0.1, RiskConfig())["maintenance_priority"] == "HIGH"
