from fastapi.testclient import TestClient

from predictmaint.service.app import create_app


def test_endpoints(client, telemetry):
    assert client.get("/health").json()["model_ready"]
    assert client.get("/model-info").status_code == 200
    assert client.post("/predict", json=telemetry).status_code == 200
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={**telemetry, "machine_type": "Z"}).status_code == 422
    assert client.post("/predict", json={**telemetry, "unknown": 1}).status_code == 422


def test_batch(client, telemetry):
    items = [
        {**telemetry, "machine_id": "a", "impact_weight": 1},
        {**telemetry, "machine_id": "b", "impact_weight": 5},
    ]
    result = client.post("/batch-predict", json={"machines": items})
    assert result.status_code == 200
    assert result.json()["machines"][0]["machine_id"] == "b"
    assert client.post("/batch-predict", json={"machines": []}).status_code == 422
    assert client.post("/batch-predict", json={"machines": [items[0]] * 101}).status_code == 422
    assert client.post("/batch-predict", json={"machines": [items[0]] * 2}).status_code == 422
    items[0]["impact_weight"] = -1
    assert client.post("/batch-predict", json={"machines": items}).status_code == 422


def test_unready(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/missing/model.joblib")
    with TestClient(create_app()) as client:
        assert not client.get("/health").json()["model_ready"]
        assert client.get("/model-info").status_code == 503


def test_inference_failure(client, predictor, telemetry, monkeypatch):
    def fail(*args):
        raise RuntimeError("private detail")

    monkeypatch.setattr(predictor, "predict", fail)
    assert client.post("/predict", json=telemetry).json() == {"detail": "Model inference failed"}
    assert (
        client.post(
            "/batch-predict", json={"machines": [{**telemetry, "machine_id": "a"}]}
        ).status_code
        == 500
    )
