"""Tests for health and readiness endpoints."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from model_serving_api.api.health import is_model_available
from model_serving_api.backends.protocol import BackendHealth
from model_serving_api.main import app
from model_serving_api.queue.manager import QueueManager


@pytest.fixture
def mock_backend():
    return AsyncMock()


@pytest.fixture
def client(mock_backend):
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.backend = mock_backend
        app.state.queue_manager = QueueManager(max_size=50)
        yield c


def test_health_always_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_healthy(client, mock_backend):
    mock_backend.health.return_value = BackendHealth(
        healthy=True,
        model_loaded=True,
        models_available=["llama3.2"],
    )

    response = client.get("/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["models_available"] == ["llama3.2"]
    assert "queue" in data
    assert data["queue"]["active"] == 0
    assert data["queue"]["max"] == 50


def test_ready_backend_unhealthy(client, mock_backend):
    mock_backend.health.return_value = BackendHealth(
        healthy=False,
        model_loaded=False,
        error="Cannot connect to Ollama",
    )

    response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert "Cannot connect" in data["reason"]


def test_ready_queue_full(client, mock_backend):
    mock_backend.health.return_value = BackendHealth(
        healthy=True,
        model_loaded=True,
        models_available=["llama3.2"],
    )
    app.state.queue_manager = QueueManager(max_size=0)

    response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert "full" in data["reason"].lower()


def test_ready_no_models_loaded(client, mock_backend):
    # Regression: this used to return 200 "ready" even though no request could succeed.
    mock_backend.health.return_value = BackendHealth(
        healthy=True,
        model_loaded=False,
        models_available=[],
    )

    response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert "llama3.2" in data["reason"]


def test_ready_configured_model_missing(client, mock_backend):
    mock_backend.health.return_value = BackendHealth(
        healthy=True,
        model_loaded=True,
        models_available=["phi:latest", "llama3:8b"],
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["models_available"] == ["phi:latest", "llama3:8b"]


@pytest.mark.parametrize(
    ("model_name", "available", "expected"),
    [
        ("llama3.2", ["llama3.2"], True),
        ("llama3.2", ["llama3.2:latest"], True),
        ("llama3.2", ["llama3.2:3b"], False),
        ("llama3.2:3b", ["llama3.2:3b"], True),
        ("llama3.2:3b", ["llama3.2:latest"], False),
        ("llama3.2", [], False),
    ],
)
def test_is_model_available(model_name, available, expected):
    assert is_model_available(model_name, available) is expected
