import pytest

from predictmaint.service.schemas import Telemetry


def test_prediction(predictor, telemetry):
    result = predictor.predict(Telemetry(**telemetry))
    assert 0 <= result["failure_probability"] <= 1
    assert len(result["top_drivers"]) == 3
    assert result["inference_latency_ms"] >= 0
    assert result["decision_threshold"] == 0.3
    assert "not causality" in result["operator_explanation"]
    assert "feature_names" in predictor.info()


@pytest.mark.parametrize(
    "key,value",
    [
        ("machine_type", "Z"),
        ("tool_wear_min", -1),
        ("torque_nm", float("nan")),
        ("rotational_speed_rpm", 0),
    ],
)
def test_schema(telemetry, key, value):
    telemetry[key] = value
    with pytest.raises(ValueError):
        Telemetry(**telemetry)
