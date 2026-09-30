"""Exercise the persisted model through real local HTTP, then stop the test server."""

import json
import os
import subprocess
import sys
import time

import httpx


def main() -> None:
    port = os.getenv("SMOKE_PORT", "8765")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "predictmaint.service.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ],
        stdout=subprocess.DEVNULL,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            deadline = time.monotonic() + 30
            while True:
                try:
                    health = client.get("/health")
                    health.raise_for_status()
                    break
                except httpx.HTTPError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("Uvicorn did not become ready") from None
                    time.sleep(0.25)
            assert health.json()["model_ready"], "Model must be trained first"
            info = client.get("/model-info")
            info.raise_for_status()
            telemetry = {
                "machine_type": "M",
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 125,
            }
            prediction = client.post("/predict", json=telemetry)
            prediction.raise_for_status()
            assert len(prediction.json()["top_drivers"]) == 3
            assert prediction.json()["top_drivers"][0]["units"] in {
                "base_model_log_odds",
                "base_model_probability",
                "constant_baseline",
            }
            items = [
                {**telemetry, "machine_id": "a", "impact_weight": 1},
                {**telemetry, "machine_id": "b", "impact_weight": 3},
            ]
            batch = client.post("/batch-predict", json={"machines": items})
            batch.raise_for_status()
            assert batch.json()["machines"][0]["machine_id"] == "b"
            invalid = client.post("/predict", json={})
            assert invalid.status_code == 422
            print(
                json.dumps(
                    {
                        "runtime_uid": os.getuid(),
                        "health": health.json(),
                        "model_name": info.json()["model_name"],
                        "decision_threshold": prediction.json()["decision_threshold"],
                        "single_status": prediction.status_code,
                        "batch_status": batch.status_code,
                        "invalid_status": invalid.status_code,
                        "single_prediction": prediction.json(),
                        "ranked_machine_ids": [
                            row["machine_id"] for row in batch.json()["machines"]
                        ],
                    },
                    indent=2,
                )
            )
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
