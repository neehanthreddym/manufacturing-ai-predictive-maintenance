import numpy as np
import pytest

from predictmaint.data.validate import validate_data


def test_valid(raw):
    df, summary = validate_data(raw)
    assert summary["rows"] == 60 and summary["target_prevalence"] == 0.25


@pytest.mark.parametrize(
    "change",
    ["missing", "target", "duplicate", "type", "numeric", "negative", "infinite", "empty_numeric"],
)
def test_invalid(raw, change):
    if change == "missing":
        raw = raw.drop(columns="Torque [Nm]")
    elif change == "target":
        raw.loc[0, "Machine failure"] = 2
    elif change == "duplicate":
        raw.loc[1, "UDI"] = raw.loc[0, "UDI"]
    elif change == "type":
        raw.loc[0, "Type"] = "Z"
    elif change == "numeric":
        raw["Torque [Nm]"] = raw["Torque [Nm]"].astype(object)
        raw.loc[0, "Torque [Nm]"] = "bad"
    elif change == "negative":
        raw.loc[0, "Tool wear [min]"] = -1
    elif change == "infinite":
        raw.loc[0, "Torque [Nm]"] = np.inf
    else:
        raw["Torque [Nm]"] = np.nan
    with pytest.raises(ValueError):
        validate_data(raw)


def test_missing_imputed_later(raw):
    raw.loc[0, "Torque [Nm]"] = np.nan
    _, s = validate_data(raw)
    assert s["missing_values"]["torque_nm"] == 1


def test_all_missing_type(raw):
    raw["Type"] = np.nan
    with pytest.raises(ValueError, match="Entirely missing machine type"):
        validate_data(raw)
