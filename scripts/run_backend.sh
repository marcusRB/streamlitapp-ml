#!/bin/bash

echo "Starting Backend API..."
cd "$(dirname "$0")/.."
uvicorn backend.api.main:app \
    --reload \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
