from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model backend
    model_name: str = "llama3.2"
    ollama_url: str = "http://ollama:11434"

    # Queue and batching
    max_queue_size: int = 50
    batch_wait_ms: int = 100
    batch_max_size: int = 8

    # Timeouts
    request_timeout_seconds: int = 60
    shutdown_grace_seconds: int = 30

    # Startup warm-up: one tiny request to load the model before serving traffic
    warmup_enabled: bool = True
    # A cold model load can take far longer than a normal request (90 s observed on a
    # laptop GPU), so the warm-up gets its own, longer timeout.
    warmup_timeout_seconds: int = 300

    # Auth
    api_key: str | None = None

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    model_config = {"env_prefix": ""}
