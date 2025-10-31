.PHONY: help install test clean pipeline figures experiments benchmark lint format check-status

PYTHON ?= python
PYTHONPATH := src:$(PYTHONPATH)
export PYTHONPATH

# Default target
help:
	@echo "PINN Options Pricing - Available Make Targets"
	@echo "=============================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install              - Install all dependencies"
	@echo "  make install-dev          - Install with dev dependencies"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test                 - Run all tests"
	@echo "  make test-coverage        - Run tests with coverage report"
	@echo "  make lint                 - Run code linters (ruff)"
	@echo "  make format               - Format code with black"
	@echo ""
	@echo "Experiments & Benchmarks:"
	@echo "  make experiments          - Run comprehensive experiments (30K epochs, ~16 min)"
	@echo "  make benchmark            - Compare all pricing methods"
	@echo "  make benchmark-quick      - Quick benchmark (fewer epochs)"
	@echo ""
	@echo "Figures & Analysis:"
	@echo "  make figures              - Generate all publication figures"
	@echo "  make pipeline             - Run full validation pipeline"
	@echo ""
	@echo "Utilities:"
	@echo "  make check-status         - Check status of running experiments"
	@echo "  make clean                - Clean generated files and caches"
	@echo "  make clean-all            - Deep clean including outputs"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install optuna pre-commit

# Testing targets
test:
	$(PYTHON) -m pytest tests/ -v

test-coverage:
	$(PYTHON) -m pytest tests/ --cov=multi_option --cov-report=html --cov-report=term

# Code quality targets
lint:
	$(PYTHON) -m ruff check src/ scripts/ tests/

format:
	$(PYTHON) -m black src/ scripts/ tests/ --line-length 100

# Experiments & Benchmarking
experiments:
	@echo "Running comprehensive experiments for publication..."
	@echo "This will train PINN for 30K epochs (~16 minutes) and generate 7 figures"
	$(PYTHON) scripts/run_comprehensive_experiments.py

benchmark:
	@echo "Running benchmark comparison of all pricing methods..."
	$(PYTHON) scripts/benchmark_methods.py

benchmark-quick:
	@echo "Running quick benchmark (fewer epochs)..."
	$(PYTHON) scripts/benchmark_methods.py --quick

# Status monitoring
check-status:
	@echo "Checking running processes..."
	@ps aux | grep -E "python.*scripts" | grep -v grep || echo "No experiments currently running"
	@echo ""
	@if [ -d output/figures ]; then \
		echo "Generated figures:"; \
		ls -lh output/figures/*.png 2>/dev/null || echo "  No figures yet"; \
	fi
	@echo ""
	@if [ -f output/comprehensive_results.json ]; then \
		echo "Latest experiment results found:"; \
		cat output/comprehensive_results.json | head -20; \
	fi

# Pipeline targets
pipeline:
	@echo "Running full validation pipeline..."
	$(PYTHON) scripts/run_full_pipeline.py --config configs/pipeline.toml

figures:
	@echo "Generating pipeline figures..."
	MPLCONFIGDIR=/tmp/mpl_fig $(PYTHON) scripts/generate_pipeline_figures.py

# Cleaning targets
clean:
	@echo "Cleaning caches and temporary files..."
	rm -rf **/__pycache__ **/*.pyc **/*.pyo **/.pytest_cache
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	rm -rf scripts/__pycache__ src/**/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cache cleaned successfully"

clean-all: clean
	@echo "Deep cleaning outputs and logs..."
	rm -rf output/*.json output/*.pt
	rm -f *.log
	rm -f pipeline_summary.json
	@echo "Note: Figures in output/figures/ are preserved (tracked in git)"
	@echo "      Archives folder preserved"
