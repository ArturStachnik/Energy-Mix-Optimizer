.PHONY: help install lint format type test train-synthetic train api docker-build clean

PY := python
PIP := pip
PKG := energy_mix_optimizer

help:
	@echo "Targets:"
	@echo "  install          Editable install with dev extras."
	@echo "  lint             Run ruff."
	@echo "  format           Apply ruff autofixes."
	@echo "  type             Run mypy."
	@echo "  test             Run the pytest suite."
	@echo "  train-synthetic  Train models on offline synthetic data."
	@echo "  train            Train models on real REE/Open-Meteo data."
	@echo "  api              Run the API locally with reload."
	@echo "  docker-build     Build the production Docker image."
	@echo "  clean            Remove caches and artifacts."

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff check --fix src tests
	ruff format src tests

type:
	mypy src

test:
	$(PY) -m pytest

train-synthetic:
	emo-train --synthetic --start 2023-01-01 --end 2023-06-30 \
	    --artifacts-dir artifacts/models --cv-splits 3

train:
	emo-train --start 2022-01-01 --end 2024-12-31 \
	    --artifacts-dir artifacts/models --cv-splits 5

api:
	uvicorn $(PKG).api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t energy-mix-optimizer:latest .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
