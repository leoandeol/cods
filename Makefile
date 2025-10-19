.PHONY: help install dev-install venv coco dotenv format lint test test-fast coverage clean build pre-commit ci

# Configuration
CODS_PYTHON ?= python3
COCO_DIR ?= ./data/coco
VAL = http://images.cocodataset.org/zips/val2017.zip
ANNOTATIONS = http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Full setup: venv + dependencies + COCO dataset"
	@echo "  make dev-install    - Install package with dev dependencies"
	@echo "  make venv           - Create virtual environment"
	@echo "  make coco           - Download COCO validation dataset"
	@echo "  make dotenv         - Create .env file with paths"
	@echo ""
	@echo "Development:"
	@echo "  make format         - Format code with ruff"
	@echo "  make lint           - Lint code with ruff"
	@echo "  make typecheck      - Run mypy type checking"
	@echo "  make pre-commit     - Install and run pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-fast      - Run tests in parallel"
	@echo "  make coverage       - Run tests with coverage report"
	@echo "  make quick-test     - Run failed tests first, stop on first failure"
	@echo ""
	@echo "Build & Clean:"
	@echo "  make build          - Build package"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make ci             - Run all CI checks locally"

# Virtual environment setup
venv:
	$(CODS_PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev]"
	@echo ""
	@echo "Virtual environment created! Activate with:"
	@echo "  source .venv/bin/activate"

# Installation
install: dotenv venv coco
	@echo ""
	@echo "Installation complete!"
	@echo "Activate environment: source .venv/bin/activate"

dev-install:
	pip install -e ".[dev]"
	pre-commit install
	@echo "Development environment ready!"

# COCO dataset download
coco:
	@echo "Downloading COCO dataset to: $(COCO_DIR)"
	@mkdir -p $(COCO_DIR)
	@cd $(COCO_DIR) && \
		wget -nc $(VAL) && \
		wget -nc $(ANNOTATIONS) && \
		unzip -q -n val2017.zip && \
		rm -f val2017.zip && \
		unzip -q -n annotations_trainval2017.zip && \
		rm -f annotations_trainval2017.zip
	@echo "COCO dataset downloaded to $(COCO_DIR)"

# Create .env file
dotenv:
	@echo "PROJECT_PATH=$$(pwd)" > .env
	@echo "COCO_PATH=$(COCO_DIR)" >> .env
	@echo ".env file created with project paths"

# Code quality
format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .

typecheck:
	mypy cods/ --ignore-missing-imports

# Testing
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -n auto

test-watch:
	pytest-watch tests/

coverage:
	pytest tests/ --cov=cods --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

quick-test:
	pytest tests/ -v -x --ff

# Pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	python -m build
	twine check dist/*

# Run all CI checks locally
ci: format lint test coverage
	@echo ""
	@echo "✓ All CI checks passed!"

# Notebook support
notebook:
	jupyter notebook

# Profile code
profile:
	python -m line_profiler -v