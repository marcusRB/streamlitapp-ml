#!/bin/bash

echo "Starting Frontend Dashboard..."
cd "$(dirname "$0")/.."
streamlit run frontend/app.py \
    --server.port 8502 \
    --server.address 0.0.0.0
