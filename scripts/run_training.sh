#!/bin/bash

echo "Running Training Pipeline..."
cd "$(dirname "$0")/.."

# Feature engineering
python -m backend.core.feature_engineering

# Model training
python -m backend.core.model_training

echo "Training complete!"