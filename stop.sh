#!/bin/bash

echo "🛑 Stopping AI Drug Repurposing Engine..."

# Check if PID files exist
if [ -f ".backend.pid" ]; then
    BACKEND_PID=$(cat .backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo "Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        echo "✅ Backend stopped"
    else
        echo "⚠️  Backend process not found"
    fi
    rm .backend.pid
else
    echo "⚠️  No backend PID file found"
fi

if [ -f ".frontend.pid" ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo "Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        echo "✅ Frontend stopped"
    else
        echo "⚠️  Frontend process not found"
    fi
    rm .frontend.pid
else
    echo "⚠️  No frontend PID file found"
fi

# Also kill any remaining processes on the ports
echo ""
echo "Checking for processes on ports 8000 and 3000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "✅ Killed remaining processes on port 8000" || echo "No processes on port 8000"
lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "✅ Killed remaining processes on port 3000" || echo "No processes on port 3000"

echo ""
echo "✅ All services stopped!"