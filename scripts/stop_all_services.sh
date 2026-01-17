#!/bin/bash

# Stop All Services Script

set -e

echo "=================================================="
echo "  CKD Detection - Stopping All Services"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Function to kill process
kill_process() {
    local pid=$1
    local name=$2
    
    if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
        echo -n "Stopping $name (PID: $pid)..."
        kill $pid 2>/dev/null || kill -9 $pid 2>/dev/null
        sleep 1
        
        if kill -0 $pid 2>/dev/null; then
            echo -e " ${RED}✗ Failed${NC}"
            return 1
        else
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
    else
        echo -e "${YELLOW}$name is not running${NC}"
        return 0
    fi
}

# Stop services by PID files
if [ -f "logs/mlflow.pid" ]; then
    MLFLOW_PID=$(cat logs/mlflow.pid)
    kill_process $MLFLOW_PID "MLflow"
    rm -f logs/mlflow.pid
fi

if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    kill_process $BACKEND_PID "Backend"
    rm -f logs/backend.pid
fi

if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    kill_process $FRONTEND_PID "Frontend"
    rm -f logs/frontend.pid
fi

# Fallback: kill by port
echo ""
echo "Checking ports..."

if lsof -ti:5000 >/dev/null 2>&1; then
    echo -n "Killing process on port 5000..."
    lsof -ti:5000 | xargs kill -9 2>/dev/null
    echo -e " ${GREEN}✓${NC}"
fi

if lsof -ti:8000 >/dev/null 2>&1; then
    echo -n "Killing process on port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    echo -e " ${GREEN}✓${NC}"
fi

if lsof -ti:8502 >/dev/null 2>&1; then
    echo -n "Killing process on port 8502..."
    lsof -ti:8502 | xargs kill -9 2>/dev/null
    echo -e " ${GREEN}✓${NC}"
fi

# Clean up
rm -f logs/services.json

echo ""
echo "=================================================="
echo -e "  ${GREEN}✅ All services stopped${NC}"
echo "=================================================="
echo ""