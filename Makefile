.PHONY: install data run test lint

install:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py

run:              ## walk-forward forecasts + backtests -> reports/ (~1.5 min)
	python -m fxrisk.pipeline

test:
	pytest -q

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts
