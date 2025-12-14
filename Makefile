.PHONY: run test lint format clean install

install:
	uv sync

run:
	uv run python src/gclear/bot.py

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src/gclear --cov-report=term-missing

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

check: lint test

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
