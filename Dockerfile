# Stage 1: Builder
FROM python:3.13-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir hatchling

COPY pyproject.toml README.md ./
RUN mkdir -p src/model_serving_api && \
    echo '__version__ = "0.1.0"' > src/model_serving_api/__init__.py

RUN pip install --no-cache-dir --prefix=/install .

COPY src/ src/
RUN pip install --no-cache-dir --no-deps --prefix=/install .

# Stage 2: Runtime
FROM python:3.13-slim AS runtime

RUN groupadd --gid 1000 serving && \
    useradd --uid 1000 --gid serving --shell /bin/bash --create-home serving

WORKDIR /app

COPY --from=builder /install /usr/local

USER serving

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "model_serving_api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "30"]
