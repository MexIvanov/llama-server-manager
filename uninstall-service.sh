#!/usr/bin/env bash
#
# uninstall-service.sh — remove the llama-server-manager systemd *user* service.
#
# Usage:
#   ./uninstall-service.sh [--purge]
#
#   --purge      Also delete the project-local .venv (if one was created).
#                Never touches conda environments.
#   -h, --help   Show this help and exit
#
# Note: lingering (loginctl enable-linger) is left enabled since other user
# services may rely on it. To disable: loginctl disable-linger "$USER"
#
set -euo pipefail

SERVICE_NAME="llama-server-manager"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${SYSTEMD_USER_UNIT_DIR:-$HOME/.config/systemd/user}"
UNIT_FILE="$UNIT_DIR/$SERVICE_NAME.service"

PURGE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --purge)   PURGE=1 ;;
        -h|--help) sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)         echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

systemctl --user disable --now "$SERVICE_NAME.service" 2>/dev/null || true
# llama-server, the RPC worker and custom scripts run in their own scopes so they survive service
# restarts; with the manager gone nothing would track them, so stop them too.
systemctl --user stop "$SERVICE_NAME-*.scope" 2>/dev/null || true
rm -f "$UNIT_FILE"
systemctl --user daemon-reload 2>/dev/null || true
systemctl --user reset-failed "$SERVICE_NAME.service" 2>/dev/null || true
echo "Removed unit file: $UNIT_FILE"

if [[ "$PURGE" -eq 1 ]]; then
    if [[ -d "$PROJECT_DIR/.venv" ]]; then
        rm -rf "$PROJECT_DIR/.venv"
        echo "Purged: $PROJECT_DIR/.venv"
    fi
fi

echo "✅ $SERVICE_NAME uninstalled."
