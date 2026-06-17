.PHONY: install test test-verbose test-file test-hook test-coverage run run-example clean build publish help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install in editable mode with dev+tui extras
	pip install -e ".[dev,tui]"

test:  ## Run all tests
	pytest

test-verbose:  ## Run all tests with verbose output
	pytest -v

test-file:  ## Run tests in a single file (usage: make test-file F=tests/test_config.py)
	pytest $(F)

test-hook:  ## Run tests matching "hook"
	pytest -k "hook"

test-coverage:  ## Run tests with coverage report
	pytest --cov=koboi --cov-report=term-missing

run:  ## Run CLI chat (usage: make run C=configs/simple_chat.yaml)
	koboi chat $(C)

run-example:  ## Run an example (usage: make run-example E=01)
	python examples/$(E)_*.py

clean:  ## Remove build artifacts and runtime files
	rm -rf build/ dist/ *.egg-info koboi_agent.egg-info
	rm -rf __pycache__ koboi/**/__pycache__ tests/**/__pycache__
	rm -rf .pytest_cache htmlcov .coverage
	rm -f koboi_memory.db koboi_memory.db-shm koboi_memory.db-wal
	rm -f koboi_trust.db .agent_memory.json

build:  ## Build sdist and wheel
	python -m build

publish: build  ## Publish to PyPI (requires TWINE_USERNAME/TWINE_PASSWORD or API token)
	twine upload dist/*
