#!/usr/bin/env bash
set -euo pipefail

# Install the systemd service for model-serving-api.
# Run on the server after the first deploy.
#
# Usage: sudo ./deploy/install-service.sh

SERVICE_FILE="/etc/systemd/system/model-serving-api.service"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "${SCRIPT_DIR}/model-serving-api.service" "${SERVICE_FILE}"
systemctl daemon-reload
systemctl enable model-serving-api.service
systemctl start model-serving-api.service
systemctl status model-serving-api.service

echo ""
echo "Service installed. Useful commands:"
echo "  journalctl -u model-serving-api -f     # Follow logs"
echo "  systemctl restart model-serving-api     # Restart"
echo "  systemctl stop model-serving-api        # Stop"
