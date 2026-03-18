.PHONY: setup dev-install test lint format jupyter docker-build docker-run docker-shell clean

# ── Standalone Setup ──────────────────────────────────────────────
setup:
	uv sync --all-extras

dev-install:
	uv sync --all-extras

# ── Development ───────────────────────────────────────────────────
test:
	uv run pytest -v

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check . --fix

jupyter:
	uv run jupyter notebook --notebook-dir=notebooks

# ── Docker ────────────────────────────────────────────────────────
docker-build:
	docker build -t rag-basics .

docker-run:
	docker compose up

docker-shell:
	docker compose run --rm rag-basics bash

# ── Cleanup ───────────────────────────────────────────────────────
clean:
	rm -rf vectordb_data/ __pycache__/ .pytest_cache/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
