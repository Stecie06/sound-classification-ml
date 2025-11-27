#!/bin/bash

# Script to scale API containers and run load tests

# Check Docker Compose version
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Docker Compose is not available."
    exit 1
fi

CONTAINERS=$1
USERS=$2
RUNTIME=${3:-120}  # Default 2 minutes
SPAWN_RATE=${4:-10}

if [ -z "$CONTAINERS" ] || [ -z "$USERS" ]; then
    echo "Usage: $0 <num_containers> <num_users> [runtime_seconds] [spawn_rate]"
    echo "Example: $0 3 100 300 20"
    echo ""
    echo "Common test scenarios:"
    echo "  Light load:  $0 1 50 120 10"
    echo "  Medium load: $0 2 100 180 15" 
    echo "  Heavy load:  $0 3 200 300 20"
    exit 1
fi

echo "🔧 Scaling to $CONTAINERS containers with $USERS users for $RUNTIME seconds"

# For simple testing without nginx, we'll use direct API connection
echo "Using direct API connection (simplified setup)..."

# Scale API containers
echo "Scaling ML API to $CONTAINERS replicas..."
$COMPOSE_CMD up -d --scale api=$CONTAINERS

echo "Waiting for containers to be ready..."
sleep 20

# Check health of scaled services
HEALTHY_COUNT=0
for i in $(seq 1 10); do
    HEALTHY_COUNT=$($COMPOSE_CMD ps api | grep "Up" | wc -l)
    if [ "$HEALTHY_COUNT" -eq "$CONTAINERS" ]; then
        break
    fi
    echo "Waiting for containers... ($HEALTHY_COUNT/$CONTAINERS healthy)"
    sleep 5
done

if [ "$HEALTHY_COUNT" -ne "$CONTAINERS" ]; then
    echo "⚠️  Not all containers are healthy. Proceeding with load test anyway..."
fi

# Get the API URL (use first container)
API_URL="http://localhost:5000"

# Run load test
echo "Starting load test with $USERS users (spawn rate: $SPAWN_RATE/s)..."
echo "Target URL: $API_URL"

$COMPOSE_CMD run --rm locust locust -f locustfile.py \
    --host $API_URL \
    --users $USERS \
    --spawn-rate $SPAWN_RATE \
    --run-time ${RUNTIME}s \
    --headless \
    --csv=load_test_results_${CONTAINERS}_containers_${USERS}_users \
    --html=load_test_report_${CONTAINERS}_containers_${USERS}_users.html

echo "📊 Load test complete!"
echo "Results saved to:"
echo "  - load_test_results_${CONTAINERS}_containers_${USERS}_users.csv"
echo "  - load_test_report_${CONTAINERS}_containers_${USERS}_users.html"

# Display quick summary
if [ -f "load_test_results_${CONTAINERS}_containers_${USERS}_users_requests.csv" ]; then
    echo ""
    echo "📈 Quick Summary:"
    # Use awk to calculate average response time
    awk -F, '
    BEGIN { total=0; count=0; }
    NR>1 {
        response_time = $4;
        if (response_time > 0) {
            total += response_time;
            count++;
        }
    }
    END {
        if (count > 0) {
            printf "  Average Response Time: %.2f ms\n", total/count;
            printf "  Total Requests: %d\n", count;
        }
    }' "load_test_results_${CONTAINERS}_containers_${USERS}_users_requests.csv"
fi

# Reset to single container
echo ""
echo "Rescaling to 1 container..."
$COMPOSE_CMD up -d --scale api=1

echo "✅ Scaling test completed!"