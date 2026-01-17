#!/bin/bash

# Complete Service Startup Script
# Starts MLflow, Backend API, and Streamlit Frontend

set -e  # Exit on error

echo "=================================================="
echo "  CKD Detection - Starting All Services"
echo "=================================================="
echo ""

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "📂 Project Root: $PROJECT_ROOT"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "⏳ Waiting for $service_name to start..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e " ${RED}✗${NC}"
    echo -e "${RED}Failed to start $service_name${NC}"
    return 1
}

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p mlruns backend/ml/models logs data/raw data/processed reports figures

# Check Python version
echo ""
echo "🐍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Virtual environment not found${NC}"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate || {
    echo -e "${RED}Failed to activate virtual environment${NC}"
    exit 1
}

# Check if requirements are installed
echo ""
echo "📦 Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

# Kill existing processes on ports
echo ""
echo "🔍 Checking for existing services..."

if check_port 5000; then
    echo -e "${YELLOW}Port 5000 is in use. Killing process...${NC}"
    lsof -ti:5000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

if check_port 8000; then
    echo -e "${YELLOW}Port 8000 is in use. Killing process...${NC}"
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

if check_port 8502; then
    echo -e "${YELLOW}Port 8502 is in use. Killing process...${NC}"
    lsof -ti:8502 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo -e "${GREEN}✓ Ports cleared${NC}"

# Start MLflow
echo ""
echo "=================================================="
echo "  1️⃣  Starting MLflow Tracking Server"
echo "=================================================="

nohup mlflow server \
    --backend-store-uri file://$PROJECT_ROOT/mlruns \
    --default-artifact-root $PROJECT_ROOT/mlruns/artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    > logs/mlflow.log 2>&1 &

MLFLOW_PID=$!
echo "MLflow PID: $MLFLOW_PID"
echo $MLFLOW_PID > logs/mlflow.pid

wait_for_service "http://localhost:5000/health" "MLflow"

echo -e "${GREEN}✓ MLflow running at: http://localhost:5000${NC}"

# Start Backend API
echo ""
echo "=================================================="
echo "  2️⃣  Starting Backend API (FastAPI)"
echo "=================================================="

cd "$PROJECT_ROOT"

nohup uvicorn backend.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    > logs/backend.log 2>&1 &

BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
echo $BACKEND_PID > logs/backend.pid

wait_for_service "http://localhost:8000/health" "Backend API"

echo -e "${GREEN}✓ Backend running at: http://localhost:8000${NC}"
echo -e "${BLUE}  📚 API Docs: http://localhost:8000/docs${NC}"

# Start Streamlit Frontend
echo ""
echo "=================================================="
echo "  3️⃣  Starting Frontend (Streamlit)"
echo "=================================================="

nohup streamlit run frontend/app.py \
    --server.port 8502 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > logs/frontend.log 2>&1 &

FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
echo $FRONTEND_PID > logs/frontend.pid

wait_for_service "http://localhost:8502/_stcore/health" "Streamlit Frontend"

echo -e "${GREEN}✓ Frontend running at: http://localhost:8502${NC}"

# Summary
echo ""
echo "=================================================="
echo "  ✅ All Services Started Successfully!"
echo "=================================================="
echo ""
echo -e "${GREEN}Services:${NC}"
echo "  🧪 MLflow:    http://localhost:5000"
echo "  🚀 Backend:   http://localhost:8000"
echo "  📚 API Docs:  http://localhost:8000/docs"
echo "  🎨 Frontend:  http://localhost:8502"
echo ""
echo -e "${BLUE}Process IDs:${NC}"
echo "  MLflow:   $MLFLOW_PID"
echo "  Backend:  $BACKEND_PID"
echo "  Frontend: $FRONTEND_PID"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo "  MLflow:   tail -f logs/mlflow.log"
echo "  Backend:  tail -f logs/backend.log"
echo "  Frontend: tail -f logs/frontend.log"
echo ""
echo -e "${YELLOW}To stop all services:${NC}"
echo "  ./scripts/stop_all_services.sh"
echo ""
echo "=================================================="
echo ""

# Save service info
cat > logs/services.json <<EOF
{
  "mlflow": {
    "pid": $MLFLOW_PID,
    "url": "http://localhost:5000",
    "port": 5000
  },
  "backend": {
    "pid": $BACKEND_PID,
    "url": "http://localhost:8000",
    "port": 8000
  },
  "frontend": {
    "pid": $FRONTEND_PID,
    "url": "http://localhost:8502",
    "port": 8502
  }
}
EOF

echo -e "${GREEN}✓ Service info saved to logs/services.json${NC}"
echo ""

# Open browser (optional)
read -p "Open services in browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Opening services in browser..."
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "http://localhost:8502"
        sleep 2
        open "http://localhost:8000/docs"
        sleep 2
        open "http://localhost:5000"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open "http://localhost:8502" 2>/dev/null || \
        x-www-browser "http://localhost:8502" 2>/dev/null
    fi
fi

echo ""
echo -e "${GREEN}🎉 Setup complete! Enjoy using CKD Detection!${NC}"
echo ""