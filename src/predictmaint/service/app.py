import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from predictmaint.logging_config import configure_logging
from predictmaint.service.predictor import Predictor
from predictmaint.service.schemas import BatchRequest, BatchResponse, Prediction, Telemetry


def create_app(predictor: Predictor | None = None) -> FastAPI:
    # Load only the working directory's .env; explicit shell settings take precedence.
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

    @asynccontextmanager
    async def lifespan(app):
        configure_logging()
        logging.info("API startup")
        if app.state.predictor is None:
            try:
                app.state.predictor = Predictor(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
            except Exception:
                logging.exception("Model unavailable; run training or verify trusted artifact")
        yield

    app = FastAPI(
        title="Manufacturing AI decision-support POC",
        description=(
            "Benchmark failure detection. Not certified for safety decisions; "
            "follow manufacturer procedures. SHAP describes attribution, "
            "not causality."
        ),
        lifespan=lifespan,
    )
    app.state.predictor = predictor

    def ready():
        if app.state.predictor is None:
            raise HTTPException(503, "Model unavailable; train or verify model artifact")
        return app.state.predictor

    @app.get("/")
    def welcome() -> dict[str, str]:
        return {
            "message": (
                "Welcome to Manufacturing AI — Predictive Maintenance "
                "& Production Risk Intelligence."
            ),
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    def health():
        return {"service": "ok", "model_ready": app.state.predictor is not None}

    @app.get("/model-info")
    def model_info():
        return ready().info()

    @app.post("/predict", response_model=Prediction)
    def predict(payload: Telemetry):
        model = ready()
        try:
            return model.predict(payload)
        except Exception:
            logging.exception("Prediction failed")
            raise HTTPException(500, "Model inference failed") from None

    @app.post("/batch-predict", response_model=BatchResponse)
    def batch(payload: BatchRequest):
        model = ready()
        if len(payload.machines) > model.config.service.max_batch_size:
            raise HTTPException(422, "Configured batch limit exceeded")
        ids = [machine.machine_id for machine in payload.machines]
        if len(set(ids)) != len(ids):
            raise HTTPException(422, "Machine IDs must be unique within a batch")
        logging.info("Batch size=%d", len(ids))
        try:
            results = [
                {
                    "machine_id": machine.machine_id,
                    "impact_weight": machine.impact_weight,
                    **model.predict(machine, machine.impact_weight),
                }
                for machine in payload.machines
            ]
            return {"machines": sorted(results, key=lambda r: r["risk_score"], reverse=True)}
        except Exception:
            logging.exception("Batch inference failed")
            raise HTTPException(500, "Model inference failed") from None

    return app


app = create_app()
