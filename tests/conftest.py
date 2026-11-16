import os

import pytest
from fastapi.testclient import TestClient

from model_serving_api.main import app

# Tests never talk to a real Ollama. Without this, every TestClient start-up runs the
# model warm-up against http://ollama:11434 and waits for DNS/connect to fail (~3-4 s each).
# Settings are read when the app starts (lifespan), so setting this at import time is enough.
os.environ.setdefault("WARMUP_ENABLED", "false")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
