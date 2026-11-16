#!/usr/bin/env bash
set -euo pipefail

# Deploy model-serving-api to a remote server via SSH.
#
# Usage:
#   ./deploy/deploy.sh user@host
#
# Prerequisites:
#   - SSH access to the server (key-based auth)
#   - Docker and Docker Compose installed on the server
#   - DNS A record pointing to the server's IP
#   - .env file on the server at /opt/model-serving-api/.env with DOMAIN=yourdomain.com

REMOTE="${1:?Usage: ./deploy/deploy.sh user@host}"
REMOTE_DIR="/opt/model-serving-api"
MODEL="llama3.2"

echo "==> Syncing project to ${REMOTE}:${REMOTE_DIR}"
rsync -az --delete \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude '.pytest_cache' \
    --exclude '.ruff_cache' \
    --exclude '__pycache__' \
    --exclude '.env' \
    . "${REMOTE}:${REMOTE_DIR}/"

echo "==> Building and starting services"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && docker compose up -d --build"

echo "==> Pulling model (if not already present)"
ssh "${REMOTE}" "docker exec model-serving-ollama ollama pull ${MODEL}" || true

echo "==> Waiting for services to start"
sleep 5

echo "==> Checking health"
ssh "${REMOTE}" "curl -sf --max-time 5 http://localhost:8000/health" \
    || echo "Not ready yet — model warm-up can take ~60s. Check: ssh ${REMOTE} 'cd ${REMOTE_DIR} && docker compose logs -f api'"

echo ""
echo "==> Done. Services running on ${REMOTE}."
echo "    Logs:   ssh ${REMOTE} 'cd ${REMOTE_DIR} && docker compose logs -f'"
echo "    Status: ssh ${REMOTE} 'cd ${REMOTE_DIR} && docker compose ps'"
