#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

cleanup() {
  echo ""
  echo "  🛑 Shutting down..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
  echo "  Done."
}
trap cleanup EXIT INT TERM

# -- Backend ---------------------------------------------------------------
echo ""
echo "  🚀 Starting FastAPI backend on http://localhost:8000"
uv run uvicorn api.main:app --reload --port 8000 &
BACKEND_PID=$!

# -- Frontend --------------------------------------------------------------
cd "$ROOT/frontend"

if [ ! -d "node_modules" ]; then
  echo "  📦 Installing frontend dependencies..."
  npm install
fi

echo "  ⚛️  Starting React frontend on http://localhost:5173"
npm run dev &
FRONTEND_PID=$!

# -- Ready -----------------------------------------------------------------
echo ""
echo "  ✅ Both servers running!"
echo "     Backend API:  http://localhost:8000/api/health"
echo "     Frontend UI:  http://localhost:5173"
echo "     Press Ctrl+C to stop both."
echo ""

wait
