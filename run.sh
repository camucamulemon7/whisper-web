#!/bin/bash
# Quick start script for Whisper Real-time Transcription

echo "🚀 Starting Whisper Real-time Transcription..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file to set your LLM API key"
fi

# Pull latest images
echo "📦 Pulling latest images from GitHub Container Registry..."
docker compose pull
docker compose build --no-cache

# Start services
echo "🎬 Starting services..."
docker compose -f docker-compose.yaml up --force-recreate --detach

# Show logs
echo "✅ Services started! Showing logs (Ctrl+C to exit logs)..."
echo "🌐 Application will be available at http://localhost:5173"
docker compose logs -f
