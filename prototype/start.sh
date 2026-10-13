#!/usr/bin/env bash
# ============================================================
# Phase 09 CDSS Prototype — Startup Script
# Installs deps (if needed) and starts the FastAPI backend.
# The frontend (prototype/frontend/index.html) is served at http://localhost:8181/
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"

echo "==================================================="
echo "  CDSS Phase 09 Prototype — Starting Backend"
echo "==================================================="

# Create virtual environment if missing
if [ ! -d "$VENV_DIR" ]; then
  echo "[1/3] Creating virtual environment…"
  python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install / upgrade dependencies
echo "[2/3] Installing dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r "$BACKEND_DIR/requirements.txt"

echo "[3/3] Starting FastAPI server…"
echo ""
echo "  → API:      http://localhost:8181"
echo "  → Frontend: http://localhost:8181/"
echo "  → API Docs: http://localhost:8181/docs"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

cd "$BACKEND_DIR"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8181
