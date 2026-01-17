#!/bin/bash

echo "Setting up CKD Detection Project..."

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
python -c "from backend.utils.config import Paths; Paths.create_directories()"

echo "Setup complete!"
echo "Start backend: ./scripts/run_backend.sh"
echo "Start frontend: ./scripts/run_frontend.sh"