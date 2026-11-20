#!/usr/bin/env bash
#
# install-service.sh — install llama-server-manager as a systemd *user* service.
#
# Runs as your user (no sudo), with native access to your home dir, conda env and
# GPU. Uses `loginctl enable-linger` so it starts at boot without an interactive
# login. The service serves the web UI on http://localhost:5000 (app.py's port).
#
# Usage:
#   ./install-service.sh [env option] [--no-linger]
#
# Environment options (pick one; if omitted you'll be prompted):
#   --venv            Create/use a project-local .venv and install requirements
#   --conda <name>    Use an existing conda env named <name>
#   --conda=<name>    (same, = form)
#   --system          Use the system Python (python3, else python)
#   --no-linger       Do not enable lingering (service won't start until you log in)
#   -h, --help        Show this help and exit
#
set -euo pipefail

SERVICE_NAME="llama-server-manager"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${SYSTEMD_USER_UNIT_DIR:-$HOME/.config/systemd/user}"
UNIT_FILE="$UNIT_DIR/$SERVICE_NAME.service"

ENV_MODE=""       # venv | conda | system
CONDA_ENV=""
NO_LINGER=0

usage() {
    sed -n '3,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() { echo "Error: $*" >&2; exit 1; }

# Print the real path of the first Python >= 3.8 on PATH (python3, then python).
# sys.executable resolves shims/wrappers (e.g. pyenv) to the actual interpreter.
find_python() {
    local cand
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 &&
           "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null; then
            "$cand" -c 'import sys; print(sys.executable)'
            return 0
        fi
    done
    return 1
}

[[ $EUID -eq 0 ]] && die "Run as your normal user (not root/sudo); this installs a per-user service."
RUN_USER="${USER:-$(id -un)}"

# ---- parse args ----------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)       ENV_MODE="venv" ;;
        --system)     ENV_MODE="system" ;;
        --conda)      ENV_MODE="conda"
                      if [[ -n "${2:-}" && "${2:-}" != -* ]]; then CONDA_ENV="$2"; shift; fi ;;
        --conda=*)    ENV_MODE="conda"; CONDA_ENV="${1#*=}" ;;
        --no-linger)  NO_LINGER=1 ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

# ---- interactive fallback ------------------------------------------------------
if [[ -z "$ENV_MODE" ]]; then
    echo "Select the Python environment to run the service with:"
    echo "  1) Local venv  (create $PROJECT_DIR/.venv and install requirements)"
    echo "  2) Conda env   (use an existing conda environment)"
    echo "  3) System Python"
    read -rp "Choice [1-3]: " choice || die "No input; pass --venv, --conda <name> or --system."
    case "$choice" in
        1) ENV_MODE="venv" ;;
        2) ENV_MODE="conda" ;;
        3) ENV_MODE="system" ;;
        *) die "Invalid choice: $choice" ;;
    esac
fi

if [[ "$ENV_MODE" == "conda" && -z "$CONDA_ENV" ]]; then
    default_env="${CONDA_DEFAULT_ENV:-base}"
    read -rp "Conda environment name [$default_env]: " CONDA_ENV \
        || die "No input; pass --venv, --conda <name> or --system."
    CONDA_ENV="${CONDA_ENV:-$default_env}"
fi

# ---- resolve the python interpreter & install deps -----------------------------
PYTHON=""
case "$ENV_MODE" in
    venv)
        if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
            BASE_PY="$(find_python)" || die "No Python 3.8+ found on PATH (tried python3, python)."
            echo "Creating venv at $PROJECT_DIR/.venv (from $BASE_PY) ..."
            if ! "$BASE_PY" -m venv "$PROJECT_DIR/.venv"; then
                rm -rf "$PROJECT_DIR/.venv"
                die "Could not create the venv. On Debian/Ubuntu install the venv package, e.g. 'sudo apt install python3-venv'."
            fi
        fi
        PYTHON="$PROJECT_DIR/.venv/bin/python"
        echo "Installing dependencies into the venv ..."
        "$PYTHON" -m pip install --upgrade pip
        "$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
        ;;
    conda)
        CONDA="${CONDA_EXE:-$(command -v conda || true)}"
        [[ -n "$CONDA" ]] || die "conda not found (not on PATH and \$CONDA_EXE is unset)."
        # Let conda resolve the env: it may live outside <base>/envs (e.g. ~/.conda/envs).
        ENV_PREFIX="$("$CONDA" run -n "$CONDA_ENV" printenv CONDA_PREFIX 2>/dev/null || true)"
        [[ -n "$ENV_PREFIX" ]] || die "conda env '$CONDA_ENV' not found."
        PYTHON="$ENV_PREFIX/bin/python"
        [[ -x "$PYTHON" ]] || PYTHON="$ENV_PREFIX/bin/python3"
        [[ -x "$PYTHON" ]] || die "conda env '$CONDA_ENV' has no Python; run: conda install -n $CONDA_ENV python"
        echo "Installing dependencies into conda env '$CONDA_ENV' ..."
        "$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
        ;;
    system)
        PYTHON="$(find_python)" || die "No Python 3.8+ found on PATH (tried python3, python)."
        if ! "$PYTHON" -c "import flask, flask_socketio" >/dev/null 2>&1; then
            echo "Installing dependencies for the system python (user site) ..."
            "$PYTHON" -m pip install --user -r "$PROJECT_DIR/requirements.txt" \
                || die "pip install failed for $PYTHON (pip missing, or the system Python is" \
                       "externally managed / PEP 668). Re-run with --venv, or install the OS" \
                       "packages, e.g. python3-flask python3-flask-socketio."
        fi
        ;;
esac

"$PYTHON" -c "import flask, flask_socketio" || die "Dependencies are not importable by $PYTHON."
echo "Using interpreter: $PYTHON"

# ---- write the unit file -------------------------------------------------------
mkdir -p "$UNIT_DIR"
cat > "$UNIT_FILE" <<EOF
[Unit]
Description=Llama Server Manager (web UI)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart="$PYTHON" "$PROJECT_DIR/app.py"
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
echo "Wrote unit file: $UNIT_FILE"

# ---- activate ------------------------------------------------------------------
systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME.service"
systemctl --user restart "$SERVICE_NAME.service"   # restart so a re-install picks up the new unit

LINGER_OK=0
if [[ "$NO_LINGER" -eq 0 ]] && command -v loginctl >/dev/null 2>&1; then
    # Enabling linger normally needs admin authorization (polkit). Try unprivileged
    # first, then a non-interactive sudo (-n never hangs on a password prompt).
    if loginctl enable-linger "$RUN_USER" 2>/dev/null; then
        LINGER_OK=1
    elif command -v sudo >/dev/null 2>&1 && sudo -n loginctl enable-linger "$RUN_USER" 2>/dev/null; then
        LINGER_OK=1
    fi
fi

echo ""
echo "✅ $SERVICE_NAME installed and started."
echo "   Web UI:   http://localhost:5000"
echo "   Status:   systemctl --user status $SERVICE_NAME"
echo "   Logs:     journalctl --user -u $SERVICE_NAME -f"
echo "   Stop:     systemctl --user stop $SERVICE_NAME"
echo "   Remove:   ./uninstall-service.sh"
if [[ "$NO_LINGER" -eq 0 && "$LINGER_OK" -eq 0 ]]; then
    echo ""
    echo "   Note: lingering not enabled (needs admin auth). The service will stop"
    echo "         when you log out. To make it persist across logout/reboot, run:"
    echo "           sudo loginctl enable-linger \"$RUN_USER\""
fi
