import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DataConfig(SettingsModel):
    raw_path: str = "data/raw/ai4i2020.csv"
    processed_path: str = "data/processed/telemetry.csv"
    target_column: str = "Machine failure"


class SplitConfig(SettingsModel):
    test_size: float = Field(default=0.15, gt=0, lt=1)
    validation_size: float = Field(default=0.15, gt=0, lt=1)

    @model_validator(mode="after")
    def valid(self):
        if self.test_size + self.validation_size >= 1:
            raise ValueError("Split fractions must leave training data")
        return self


class ModelConfig(SettingsModel):
    target_metric: Literal["average_precision"] = "average_precision"
    cv_folds: int = Field(default=3, ge=2, le=10)


class ThresholdConfig(SettingsModel):
    strategy: Literal["f2"] = "f2"


class RiskConfig(SettingsModel):
    low_max: float = 0.25
    medium_max: float = 0.50
    high_max: float = 0.75

    @model_validator(mode="after")
    def valid(self):
        if not 0 < self.low_max < self.medium_max < self.high_max < 1:
            raise ValueError("Risk cutoffs must strictly increase within (0,1)")
        return self


class ExplanationConfig(SettingsModel):
    mode: Literal["template", "groq", "openai"] = "template"


class ServiceConfig(SettingsModel):
    max_batch_size: int = Field(default=100, ge=1, le=10000)


class Config(SettingsModel):
    random_seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    risk_policy: RiskConfig = Field(default_factory=RiskConfig)
    explanation: ExplanationConfig = Field(default_factory=ExplanationConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.getenv("CONFIG_PATH", "configs/model.yaml"))
    try:
        content = yaml.safe_load(path.read_text())
        if not isinstance(content, dict):
            raise ValueError("Expected YAML mapping")
        return Config.model_validate(content)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"Invalid configuration {path}: {exc}") from exc
