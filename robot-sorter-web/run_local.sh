#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=========================================================="
echo "  LANZANDO SO-101 WORKSTATION EN LOCAL (PYTHON)          "
echo "=========================================================="

# Check for python environments
PYTHON_BIN="python3"
if [ -f "/home/oier/.local/share/SenseCraft Robotics/r/afb1f21fdf9484b9/runtime/bin/python" ]; then
    PYTHON_BIN="/home/oier/.local/share/SenseCraft Robotics/r/afb1f21fdf9484b9/runtime/bin/python"
fi

# Check if port 8080 is free, otherwise use 8000
PORT=8080
if ss -tuln | grep -q ":8080 "; then
    PORT=8000
fi

echo "[*] Usando Python: $PYTHON_BIN"
echo "[*] Servidor disponible en: http://localhost:$PORT"
"$PYTHON_BIN" -m uvicorn server.main:app --host 0.0.0.0 --port $PORT

