from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Telemetry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    machine_type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(gt=0, le=2000)
    process_temperature_k: float = Field(gt=0, le=2000)
    rotational_speed_rpm: float = Field(gt=0, le=1000000)
    torque_nm: float = Field(gt=0, le=1000000)
    tool_wear_min: float = Field(ge=0, le=1000000)


class Machine(Telemetry):
    machine_id: str = Field(min_length=1, max_length=128)
    impact_weight: float = Field(default=1.0, ge=0, le=1000000)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    machines: list[Machine] = Field(min_length=1, max_length=10000)


class Driver(BaseModel):
    feature: str
    value: str | float
    contribution: float
    units: str


class Prediction(BaseModel):
    failure_probability: float = Field(ge=0, le=1)
    predicted_failure: bool
    decision_threshold: float
    risk_score: float
    risk_band: str
    maintenance_priority: str
    top_drivers: list[Driver]
    operator_explanation: str
    inference_latency_ms: float


class RankedPrediction(Prediction):
    machine_id: str
    impact_weight: float


class BatchResponse(BaseModel):
    machines: list[RankedPrediction]
