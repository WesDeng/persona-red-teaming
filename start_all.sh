#!/bin/bash

# Persona Red-Teaming UI - Start All Services

echo "=========================================="
echo "Persona Red-Teaming UI Startup"
echo "=========================================="
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with your OPENAI_API_KEY"
    echo "Example: cp .env.example .env"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)
echo "✓ Loaded environment variables"

# Check for Python dependencies
if ! python -c "import fastapi" 2>/dev/null; then
    echo "⚠ FastAPI not found. Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Check for Node dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo "⚠ Node modules not found. Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

echo ""
echo "Starting services..."
echo "Backend will run on: http://localhost:8000"
echo "Frontend will run on: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=========================================="
echo ""

# Start backend in background
python api/server.py &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 3

# Start frontend in background
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Trap Ctrl+C and cleanup
trap cleanup INT TERM

# Wait for both processes
wait
