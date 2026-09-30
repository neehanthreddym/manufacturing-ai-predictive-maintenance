# Verification — September 30, 2026

- Python 3.12.5; project dependencies installed successfully. Both editable and normal wheel builds succeeded. This sandbox did not process editable `.pth` imports consistently, so Make targets explicitly export `PYTHONPATH=src`; manual setup documents the same export.
- Official UCI ZIP downloaded programmatically. Validated 10,000 records, 339 failures, no missing values or duplicate identifiers. Original CSV SHA-256 is in artifact metadata.
- Download, prepare, quick training and stored evaluation stages executed successfully. Full standard search is implemented; this session's published real-dataset results are from quick mode.
- Four validation candidates compared. XGBoost selected by validation average precision; training-only sigmoid calibration retained based on validation Brier score. Threshold selected only on validation.
- Six actual evaluation/explainability figures generated. Calibration and SHAP plots visually inspected. Raw validation and held-out test metrics are available in the JSON/CSV reports.
- `make test`: **47 passed**, **99.09%** core-package coverage. Tests are offline and include fixture-only end-to-end training, artifact loading and API prediction. Four upstream deprecation warnings (Starlette/SHAP), no test failures.
- `make lint`: Ruff check and format check both passed.
- Application import verified. Temporary Uvicorn HTTP smoke verified health/readiness, model info, single scoring, fleet ranking and invalid-input 422 on the real persisted model. Evidence: `local_api_smoke.json`.
- Final `docker build -t predictmaint-ai .` succeeded using `python:3.12-slim` and dependency constraints. Docker container HTTP smoke passed with **UID 1000**, confirming non-root runtime. Evidence: `docker_api_smoke.json`. Local and container sample probabilities matched exactly. Test servers and temporary container were stopped/removed.
- Docker dependency layers precede source installation for caching. The standard Linux XGBoost distribution includes an NVIDIA NCCL dependency even though this project uses CPU histogram training/inference; no GPU requirement is introduced.
- GitHub Actions workflow added; its local Ruff/pytest equivalents pass. Hosted GitHub CI has not run because this directory has not been published.
- Git initialization was attempted but blocked by the workspace policy; its escalation request was declined. Files remain a complete local project without Git metadata.
- Default deterministic explanation and mocked OpenAI success/failure paths tested. No paid/external LLM request was executed. `.env` is not auto-loaded; optional credentials require explicit environment exports.

## Interpretation
At the validation-selected operating threshold, test recall improved from 86.27% to 90.20% compared with 0.5, with false negatives decreasing from 7 to 5. False positives increased from 12 to 47, precision fell from 78.57% to 49.46%, and test F2 decreased from 0.8462 to 0.7744. This is a real alert-volume tradeoff; optimizing validation F2 does not guarantee better test F2. The test set must not be reused for tuning.

All published ML results use the official synthetic UCI benchmark, never the test fixture. No industrial deployment, future failure horizon, aerospace validation, maintenance savings or safety certification is claimed.

## Groq provider update
Groq is now the primary optional live summary provider (`EXPLANATION_MODE=groq`, `GROQ_API_KEY`, `GROQ_MODEL`), using its official OpenAI-compatible Chat Completions endpoint. OpenAI remains an explicitly selected alternative; providers never borrow each other's credentials or automatically fail over to each other. Offline template mode remains the YAML default. No new dependencies or model retraining were needed.

Updated verification: **53 tests passed, 99.28% coverage**, Ruff lint/format passed. Mocked tests verify both endpoint/credential routes, summary-only behavior, missing credentials/models, failed calls, malformed/empty responses, and Groq's isolation from OpenAI credentials. No live Groq or OpenAI request was performed; live verification requires provider credentials and an available model.
