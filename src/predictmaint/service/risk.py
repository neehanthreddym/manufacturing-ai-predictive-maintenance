import math

from predictmaint.config import RiskConfig


def risk_decision(
    probability: float, threshold: float, policy: RiskConfig, impact_weight: float = 1.0
) -> dict:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be within [0,1]")
    if not math.isfinite(impact_weight) or impact_weight < 0:
        raise ValueError("Impact weight must be finite and nonnegative")
    band = next(
        (
            name
            for cutoff, name in [
                (policy.low_max, "LOW"),
                (policy.medium_max, "MEDIUM"),
                (policy.high_max, "HIGH"),
            ]
            if probability < cutoff
        ),
        "CRITICAL",
    )
    predicted = probability >= threshold
    priority = "HIGH" if predicted and band in ["LOW", "MEDIUM"] else band
    return {
        "failure_probability": probability,
        "predicted_failure": predicted,
        "decision_threshold": threshold,
        "risk_score": probability * impact_weight,
        "risk_band": band,
        "maintenance_priority": priority,
    }
