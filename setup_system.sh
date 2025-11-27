#!/bin/bash

echo "🔧 Setting up Sound Classification ML System"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if docker compose version &> /dev/null; then
    echo "✅ Docker Compose V2 detected"
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose V1 detected" 
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Docker Compose not found. Installing..."
    # Try to install docker-compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    COMPOSE_CMD="docker-compose"
    
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Failed to install docker-compose. Please install manually."
        exit 1
    fi
fi

echo "✅ Docker and Docker Compose are ready!"

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/upload data/temp models load_test

# Check if model exists
if [ ! -f "models/sound_classifier.pkl" ]; then
    echo "⚠️  Model file not found. Please ensure models/sound_classifier.pkl exists."
    echo "You can create it by running: cd notebook && python create_compatible_model.py"
fi

# Check if test audio exists
if [ ! -f "data/test/test_audio.wav" ]; then
    echo "⚠️  Test audio file not found. Creating a dummy file for testing..."
    mkdir -p data/test
    # Create a minimal WAV file header (dummy content)
    echo "RIFF____WAVEfmt ____________________data____" > data/test/test_audio.wav
    echo "✅ Created dummy test audio file"
fi

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   ./deploy.sh      # Deploy the system"
echo "   ./scale_test.sh 1 50 120  # Run load test with 1 container, 50 users"