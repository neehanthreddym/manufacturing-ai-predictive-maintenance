PYTHON ?= .venv/bin/python
export PYTHONPATH := src
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
RUFF = .venv/bin/ruff
.PHONY: setup data prepare train evaluate test lint format serve docker-build docker-run clean
setup:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -c requirements-lock.txt -e ".[dev]"
data:
	$(PYTHON) scripts/download_data.py
prepare:
	$(PYTHON) scripts/prepare_data.py
train:
	$(PYTHON) scripts/train_model.py $(ARGS)
evaluate:
	$(PYTHON) scripts/evaluate_model.py
test:
	$(PYTHON) -m pytest -q --cov=predictmaint --cov-report=term-missing --cov-report=xml --cov-fail-under=75
lint:
	$(RUFF) check .
	$(RUFF) format --check .
format:
	$(RUFF) check --fix .
	$(RUFF) format .
serve:
	$(PYTHON) -m uvicorn predictmaint.service.app:app --host 0.0.0.0 --port 8000
docker-build:
	docker build -t predictmaint-ai .
docker-run:
	docker run --rm -p 8000:8000 predictmaint-ai
clean:
	$(PYTHON) scripts/clean.py
