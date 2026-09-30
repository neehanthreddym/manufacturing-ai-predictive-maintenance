import logging

import numpy as np
import pandas as pd

FEATURES = [
    "machine_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
NUMERIC = FEATURES[1:]
EXCLUDED = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
RENAME = dict(
    zip(
        [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ],
        FEATURES,
    )
)
BOUNDS = {name: (0, 2000 if "temperature" in name else 1000000) for name in NUMERIC}


def validate_data(
    frame: pd.DataFrame, target: str = "Machine failure"
) -> tuple[pd.DataFrame, dict]:
    df = frame.rename(columns=RENAME).copy()
    missing = set(FEATURES + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty or df[target].isna().any() or not df[target].isin([0, 1]).all():
        raise ValueError("Target must contain binary, nonmissing values")
    if df.machine_type.notna().sum() == 0:
        raise ValueError("Entirely missing machine type")
    if not df.machine_type.dropna().isin(["L", "M", "H"]).all():
        raise ValueError("Unsupported machine type")
    for name in NUMERIC:
        values = pd.to_numeric(df[name], errors="raise")
        if values.notna().sum() == 0:
            raise ValueError(f"Entirely missing telemetry: {name}")
        low, high = BOUNDS[name]
        invalid = values.notna() & (~np.isfinite(values) | (values < low) | (values > high))
        if name != "tool_wear_min":
            invalid |= values.notna() & (values == 0)
        if invalid.any():
            raise ValueError(f"Invalid telemetry: {name}")
        df[name] = values
    duplicates = {c: int(df[c].duplicated().sum()) for c in ["UDI", "Product ID"] if c in df}
    if any(duplicates.values()) or df.duplicated().any():
        raise ValueError("Duplicate rows/identifiers; resolve before splitting")
    summary = {
        "rows": len(df),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_identifiers": duplicates,
        "target_prevalence": float(df[target].mean()),
        "excluded_columns": [c for c in EXCLUDED if c in df],
        "features": FEATURES,
    }
    logging.info("Data validation: %s", summary)
    return df, summary


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rename(columns=RENAME).loc[:, FEATURES].copy()
    assert not set(EXCLUDED) & set(result.columns)
    return result
