import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictmaint.config import Config
from predictmaint.data.validate import NUMERIC, feature_matrix, validate_data


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC,
            ),
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                categories=[["L", "M", "H"]],
                                handle_unknown="error",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                ["machine_type"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_data(config: Config) -> pd.DataFrame:
    path = Path(config.data.raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing at {path}; run make data")
    df, summary = validate_data(pd.read_csv(path), config.data.target_column)
    processed = feature_matrix(df)
    processed["target"] = df[config.data.target_column]
    Path(config.data.processed_path).parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(config.data.processed_path, index=False)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_validation.json").write_text(json.dumps(summary, indent=2))
    return processed
