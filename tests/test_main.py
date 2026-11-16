from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from model_serving_api import __version__
from model_serving_api.main import app
from model_serving_api.queue.batcher import BatchDispatcher
from model_serving_api.queue.manager import QueueManager


def _patched_backend(monkeypatch, constructor_calls: list | None = None):
    """Replace OllamaBackend in main with a mock so lifespan never touches the network."""
    backend = AsyncMock()

    def _factory(*args, **kwargs):
        if constructor_calls is not None:
            constructor_calls.append(kwargs)
        return backend

    monkeypatch.setattr("model_serving_api.main.OllamaBackend", _factory)
    return backend


def test_warmup_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("WARMUP_ENABLED", "false")
    backend = _patched_backend(monkeypatch)

    with TestClient(app):
        pass

    backend.generate.assert_not_called()


def test_warmup_runs_when_enabled(monkeypatch):
    monkeypatch.setenv("WARMUP_ENABLED", "true")
    monkeypatch.setenv("MODEL_NAME", "llama3.2:3b")
    backend = _patched_backend(monkeypatch)

    with TestClient(app):
        pass

    backend.generate.assert_awaited_once()
    warmup_request = backend.generate.await_args.args[0]
    assert warmup_request.model == "llama3.2:3b"
    assert warmup_request.max_tokens == 1


def test_warmup_uses_its_own_longer_timeout(monkeypatch):
    # Regression: warm-up shared the 60 s request timeout, so a cold model load (90 s
    # observed) always failed and the first real requests returned 504.
    monkeypatch.setenv("WARMUP_ENABLED", "true")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("WARMUP_TIMEOUT_SECONDS", "300")
    calls: list[dict] = []
    backend = _patched_backend(monkeypatch, calls)

    with TestClient(app):
        pass

    timeouts = sorted(call["timeout"] for call in calls)
    assert timeouts == [60.0, 300.0]  # serving client + dedicated warm-up client
    # The warm-up client is closed after use, plus the serving client at shutdown.
    assert backend.close.await_count == 2


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "model-serving-api"
    assert data["version"] == __version__


# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------


def test_auth_rejects_without_key():
    mock_backend = AsyncMock()
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.backend = mock_backend
        app.state.queue_manager = QueueManager(max_size=50)
        app.state.batch_dispatcher = BatchDispatcher(
            backend=mock_backend, max_batch_size=1, wait_ms=0
        )
        app.state.settings.api_key = "test-secret-key"

        response = c.post(
            "/v1/chat/completions",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_api_key"

        app.state.settings.api_key = None


def test_auth_accepts_valid_key():
    mock_backend = AsyncMock()
    mock_backend.generate.return_value = AsyncMock(
        model="llama3.2",
        content="hi",
        finish_reason="stop",
        prompt_tokens=5,
        completion_tokens=1,
        total_duration_ns=100_000_000,
    )
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.backend = mock_backend
        app.state.queue_manager = QueueManager(max_size=50)
        app.state.batch_dispatcher = BatchDispatcher(
            backend=mock_backend, max_batch_size=1, wait_ms=0
        )
        app.state.settings.api_key = "test-secret-key"

        response = c.post(
            "/v1/chat/completions",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer test-secret-key"},
        )

        assert response.status_code == 200

        app.state.settings.api_key = None


def test_auth_allows_public_paths():
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.settings.api_key = "test-secret-key"

        assert c.get("/").status_code == 200
        assert c.get("/health").status_code == 200
        assert c.get("/metrics").status_code == 200

        app.state.settings.api_key = None


def test_auth_disabled_when_no_key(client):
    app.state.settings.api_key = None

    response = client.post(
        "/v1/chat/completions",
        json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
    )

    # Should not be 401 — it'll fail for other reasons but not auth
    assert response.status_code != 401


def _get_metric_value(metrics_text: str, metric_name: str, labels: dict | None = None) -> str:
    """Extract a metric value from Prometheus text format."""
    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        search = f"{metric_name}{{{label_str}}}"
    else:
        search = metric_name

    for line in metrics_text.split("\n"):
        if line.startswith("#"):
            continue
        if search in line:
            return line.split()[-1]
    return ""


def test_error_counter_auth_failure():
    mock_backend = AsyncMock()
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.backend = mock_backend
        app.state.queue_manager = QueueManager(max_size=50)
        app.state.batch_dispatcher = BatchDispatcher(
            backend=mock_backend, max_batch_size=1, wait_ms=0
        )
        app.state.settings.api_key = "test-secret-key"

        c.post(
            "/v1/chat/completions",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
        )
        metrics = c.get("/metrics").text

        value = _get_metric_value(metrics, "inference_errors_total", {"error_type": "auth_failure"})
        assert float(value) >= 1.0

        app.state.settings.api_key = None
