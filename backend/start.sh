#!/bin/bash
set -e

echo "Starting Python query server on port ${PYTHON_QUERY_PORT:-5001}..."
python /app/python/query_server.py &
PYTHON_PID=$!

# Wait for Python server to be ready (up to 30s)
for i in $(seq 1 30); do
  if curl -sf http://localhost:${PYTHON_QUERY_PORT:-5001}/health > /dev/null 2>&1; then
    echo "Python query server ready (PID $PYTHON_PID)"
    break
  fi
  if ! kill -0 "$PYTHON_PID" 2>/dev/null; then
    echo "ERROR: Python query server exited prematurely"
    exit 1
  fi
  sleep 1
done

echo "Starting Node.js server on port ${PORT:-5000}..."
exec node server.js
