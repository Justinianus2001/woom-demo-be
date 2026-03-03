#!/bin/bash
set -e

echo "Starting Woom Audio Mixer deployment for Ubuntu server..."

# 1. Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo "Please install it by running: sudo apt update && sudo apt install docker.io -y"
    exit 1
fi

# 2. Build the Docker image
echo "📦 Building the Docker image..."
docker build -t woom-mixer .

# 3. Stop existing container if it's already running
if [ "$(docker ps -q -f name=woom-mixer-container)" ]; then
    echo "🔄 Stopping existing container..."
    docker stop woom-mixer-container
    docker rm woom-mixer-container
elif [ "$(docker ps -aq -f status=exited -f name=woom-mixer-container)" ]; then
    docker rm woom-mixer-container
fi

# 4. Run the Docker container on port 8000, and ensure it restarts automatically on server reboot
echo "🚀 Running the Docker container on port 8000..."
docker run -d --restart unless-stopped --name woom-mixer-container -p 8000:8000 woom-mixer

echo "✅ Docker container successfully deployed and running on port 8000!"
echo "If you haven't already, please configure Nginx to route traffic to localhost:8000."
