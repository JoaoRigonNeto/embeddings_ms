#!/bin/bash
set -e

export PYTHONPATH="${PYTHONPATH}:/app"

echo "Running startup tests..."
# pytest -s test/
pytest test/

echo "Tests passed. Starting server..."
uvicorn main:app --host 0.0.0.0 --port 8000
