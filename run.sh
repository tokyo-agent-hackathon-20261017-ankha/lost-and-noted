#!/usr/bin/env sh
# Lost and Noted — local server for hotel staff.
# Usage: AIAND_API_KEY=sk-... ./run.sh   (then open http://localhost:8000)
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
