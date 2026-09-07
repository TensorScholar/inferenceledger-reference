.PHONY: verify tests experiments evidence examples

verify: tests experiments evidence

tests:
	cd reference && python3 -m pytest tests/ -q
	python3 -m pytest tests/ -q

experiments:
	python3 experiments/run_experiment.py

evidence:
	python3 scripts/verify_evidence.py

examples:
	python3 examples/quickstart.py
