#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"
FRONTEND_HOST="127.0.0.1"
FRONTEND_PORT="5173"
BACKEND_PID=""
FRONTEND_PID=""
CLEANUP_DONE=0
SETUP_MODE=0

for arg in "$@"; do
  case "$arg" in
    --setup)
      SETUP_MODE=1
      ;;
    --help|-h)
      echo "Usage: bash start.sh [--setup]"
      echo "  --setup   Run dependency setup (uv sync + npm install) before starting"
      exit 0
      ;;
  esac
done

log() {
  echo "[moe] $1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[moe] Missing required command: $1"
    exit 1
  fi
}

cleanup() {
  if [ "$CLEANUP_DONE" -eq 1 ]; then
    return
  fi
  CLEANUP_DONE=1
  echo
  log "Stopping services..."
  if [ -n "$FRONTEND_PID" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  log "Stopped."
}

wait_for_backend() {
  local retries=60
  local url="http://${BACKEND_HOST}:${BACKEND_PORT}/api/health"

  for _ in $(seq 1 "$retries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done

  return 1
}

trap cleanup EXIT INT TERM

require_cmd uv
require_cmd npm
require_cmd curl

cd "$ROOT_DIR"

if [ ! -f ".env" ]; then
  log "Warning: .env not found in project root. API key auto-load may fail."
fi

if [ "$SETUP_MODE" -eq 1 ]; then
  log "Setup mode enabled: syncing Python dependencies (uv sync)..."
  UV_LINK_MODE="${UV_LINK_MODE:-copy}" uv sync

  log "Setup mode enabled: installing frontend dependencies..."
  npm install --prefix frontend --no-audit --no-fund
else
  log "Skipping dependency installation (fast start mode)."
  log "Use 'bash start.sh --setup' after dependency or lockfile changes."

  if [ ! -d ".venv" ]; then
    log "Python env not found (.venv). Run: bash start.sh --setup"
    exit 1
  fi

  if [ ! -d "frontend/node_modules" ]; then
    log "frontend/node_modules not found. Run: bash start.sh --setup"
    exit 1
  fi
fi

log "Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT} ..."
uv run --no-sync uvicorn api.main:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
BACKEND_PID=$!

log "Waiting for backend health endpoint..."
if ! wait_for_backend; then
  log "Backend did not become ready at http://${BACKEND_HOST}:${BACKEND_PORT}/api/health"
  exit 1
fi

log "Starting frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT} ..."
npm run dev --prefix frontend -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

echo
log "Services are running"
log "Frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
log "Backend : http://${BACKEND_HOST}:${BACKEND_PORT}/api/health"
log "Press Ctrl+C to stop"
echo

wait -n "$BACKEND_PID" "$FRONTEND_PID"
