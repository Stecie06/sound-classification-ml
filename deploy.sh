#!/bin/bash

echo "🚀 Deploying Sound Classification ML System to Cloud"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available (V1 or V2)
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
    echo "✅ Using Docker Compose V2"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo "✅ Using Docker Compose V1"
else
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

echo "Building Docker containers..."
$COMPOSE_CMD build

echo "Starting services..."
$COMPOSE_CMD up -d

echo "Waiting for services to be healthy..."
sleep 30

# Check health of all services
echo "Checking service health..."

# Check API health
API_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health 2>/dev/null || echo "000")
if [ "$API_HEALTH" = "200" ]; then
    echo "✅ API is healthy"
else
    echo "❌ API health check failed (Status: $API_HEALTH)"
fi

# Check Streamlit health (it might not have a health endpoint, so we check if port is open)
if nc -z localhost 8501 2>/dev/null; then
    echo "✅ Streamlit UI is running"
else
    echo "❌ Streamlit UI is not accessible"
fi

# Check Locust
if nc -z localhost 8089 2>/dev/null; then
    echo "✅ Locust is running"
else
    echo "❌ Locust is not accessible"
fi

echo ""
echo "🎉 Deployment complete!"
echo "📊 API: http://localhost:5000"
echo "🌐 Web UI: http://localhost:8501"
echo "🐜 Load Test: http://localhost:8089"
echo ""
echo "Quick test commands:"
echo "  curl http://localhost:5000/health"
echo "  curl http://localhost:5000/"