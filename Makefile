PYTHON ?= .venv/bin/python

.PHONY: test lint typecheck check

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src

check: lint typecheck test
