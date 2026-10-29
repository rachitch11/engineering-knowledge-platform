#!/usr/bin/env bash
# Start FastAPI backend and Streamlit frontend for local development (macOS).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f "app/main.py" || ! -f "streamlit_app.py" ]]; then
  echo "Error: run this script from the repository root:"
  echo "  $ROOT"
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "Error: .venv not found. Create it first:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

if ! python -c "import fastapi, streamlit, httpx" >/dev/null 2>&1; then
  echo "Error: missing dependencies. Run: pip install -r requirements.txt"
  exit 1
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8000}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${STREAMLIT_PORT:-8501}"

port_in_use() {
  local port="$1"
  if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

if port_in_use "${API_PORT}"; then
  echo "Error: port ${API_PORT} is already in use. Stop the existing server or set API_PORT."
  exit 1
fi

if port_in_use "${UI_PORT}"; then
  echo "Error: port ${UI_PORT} is already in use. Stop the existing Streamlit app or set STREAMLIT_PORT."
  exit 1
fi

echo "Checking LLM configuration..."
if [[ -z "${GEMINI_API_KEY:-}" && "${LLM_PROVIDER:-gemini}" == "gemini" ]]; then
  echo "Warning: GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
  echo "  https://aistudio.google.com/apikey"
elif [[ "${LLM_PROVIDER:-gemini}" == "ollama" ]]; then
  if ! curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
    echo "Warning: Ollama is not reachable (optional when LLM_PROVIDER=ollama)."
  fi
fi

UVICORN_PID=""
STREAMLIT_PID=""

cleanup() {
  echo
  echo "Stopping services..."
  if [[ -n "${STREAMLIT_PID}" ]] && kill -0 "${STREAMLIT_PID}" 2>/dev/null; then
    kill "${STREAMLIT_PID}" 2>/dev/null || true
  fi
  if [[ -n "${UVICORN_PID}" ]] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    kill "${UVICORN_PID}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  echo "Done."
}

trap cleanup EXIT INT TERM

echo "Starting FastAPI on http://${API_HOST}:${API_PORT}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
uvicorn app.main:app --host "${API_HOST}" --port "${API_PORT}" --workers "${UVICORN_WORKERS}" &
UVICORN_PID=$!

sleep 2
if ! curl -sf "http://${API_HOST}:${API_PORT}/health" >/dev/null; then
  echo "Error: backend failed health check on http://${API_HOST}:${API_PORT}/health"
  exit 1
fi

echo "Starting Streamlit on http://localhost:${UI_PORT}"
python -m streamlit run streamlit_app.py \
  --server.port "${UI_PORT}" \
  --server.address localhost &
STREAMLIT_PID=$!

echo
echo "Backend:  http://${API_HOST}:${API_PORT}"
echo "Frontend: http://localhost:${UI_PORT}"
echo "Press Ctrl+C to stop both services."
echo

wait "${UVICORN_PID}" "${STREAMLIT_PID}"
