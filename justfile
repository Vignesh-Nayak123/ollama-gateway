default:
    @just --list

# Setup
setup:
    uv sync --extra dev

# Development
dev:
    uv run uvicorn model_serving_api.main:app --reload

# Code quality
lint:
    uv run ruff check .
    uv run ruff format --check .

fmt:
    uv run ruff format .

# Testing
test:
    uv run pytest tests/ -v

check: lint test

# Load testing (needs the API running; see README "Load testing")
loadtest host="http://localhost:8000" users="5" duration="60s":
    uv run --extra loadtest locust -f loadtest/locustfile.py --headless         --host {{host}} -u {{users}} -r {{users}} -t {{duration}}

sweep out host="http://localhost:8000" users="1,4,8,16" duration="60s":
    uv run --extra loadtest python loadtest/sweep.py --host {{host}}         --users {{users}} --duration {{duration}} --out {{out}}

# Docker
docker-up:
    docker compose up -d

docker-down:
    docker compose down

docker-logs:
    docker compose logs -f api

# Cleanup
clean:
    rm -rf .venv __pycache__ .pytest_cache .ruff_cache dist/ build/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
