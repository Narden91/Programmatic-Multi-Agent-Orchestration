#!/usr/bin/env bash
# Single launcher for this project (WSL/Linux): cleans, recreates, starts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PID=""
FRONTEND_PID=""

export UV_PROJECT_ENVIRONMENT=".venv-wsl"
export UV_LINK_MODE="copy"

log() {
  echo "  $1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

cleanup() {
  echo ""
  log "🛑 Shutting down..."
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "$FRONTEND_PID" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  log "Done."
}

trap cleanup EXIT INT TERM

require_cmd uv
require_cmd npm

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  local elapsed=0

  while ! (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; do
    sleep 0.2
    elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge $((timeout_s * 5)) ]; then
      echo "Timed out waiting for ${host}:${port}"
      exit 1
    fi
  done
}

cd "$ROOT"

log "🧹 Cleaning backend env (.venv-wsl)..."
rm -rf .venv-wsl

log "📦 Recreating backend env with uv sync..."
uv sync

log "🧹 Cleaning frontend deps..."
rm -rf frontend/node_modules frontend/package-lock.json

log "📦 Reinstalling frontend deps..."
npm install --prefix frontend --no-audit --no-fund

log "🚀 Starting FastAPI backend on http://localhost:8000"
uv run --no-sync uvicorn api.main:app --reload --port 8000 &
BACKEND_PID=$!

log "⏳ Waiting for backend readiness..."
wait_for_port 127.0.0.1 8000 30

log "⚛️  Starting React frontend on http://localhost:5173"
npm run dev --prefix frontend &
FRONTEND_PID=$!

echo ""
log "✅ Both servers running!"
log "🌐 Open http://localhost:5173"
log "Backend API: http://localhost:8000/api/health"
log "Press Ctrl+C to stop both."
echo ""

wait -n "$BACKEND_PID" "$FRONTEND_PID"
