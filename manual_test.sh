#!/bin/bash

echo "🧪 Manual Testing for Sound Classification System"

# Check if services are running
echo "1. Checking if services are running..."
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✅ API is running"
else
    echo "❌ API is not running. Run ./deploy.sh first."
    exit 1
fi

if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Streamlit UI is running" 
else
    echo "⚠️  Streamlit UI might not be accessible"
fi

# Test API endpoints
echo ""
echo "2. Testing API endpoints..."

echo "   Home endpoint:"
curl -s http://localhost:5000/ | jq '.'

echo "   Health endpoint:"
curl -s http://localhost:5000/health | jq '.'

echo "   Model info:"
curl -s http://localhost:5000/model_info | jq '.'

# Test with a simple file if available
echo ""
echo "3. Testing prediction (if test file exists)..."
if [ -f "data/test/test_audio.wav" ]; then
    echo "   Sending test audio for prediction..."
    RESPONSE=$(curl -s -F "file=@data/test/test_audio.wav" http://localhost:5000/predict)
    if echo "$RESPONSE" | jq -e '.predicted_class' > /dev/null 2>&1; then
        echo "✅ Prediction successful!"
        echo "$RESPONSE" | jq '.'
    else
        echo "❌ Prediction failed"
        echo "$RESPONSE"
    fi
else
    echo "⚠️  No test audio file found at data/test/test_audio.wav"
fi

echo ""
echo "🎯 Next steps:"
echo "   Visit http://localhost:8501 for the web interface"
echo "   Visit http://localhost:8089 for load testing"
echo "   Run ./scale_test.sh 1 50 120 for a basic load test"