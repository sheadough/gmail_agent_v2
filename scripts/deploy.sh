#!/bin/bash
# scripts/deploy.sh - Production deployment script

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-production}
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="./logs/deployment_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 AI Agent Deployment Script"
echo "Environment: $ENVIRONMENT"
echo "Backup Directory: $BACKUP_DIR"
echo "Log File: $LOG_FILE"
echo "================================="

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$BACKUP_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to handle errors
handle_error() {
    log "❌ Error on line $1"
    log "Deployment failed. Check logs for details."
    exit 1
}

trap 'handle_error $LINENO' ERR

# Step 1: Pre-deployment backup
log "📦 Creating pre-deployment backup..."
if [ -d "./data" ]; then
    cp -r ./data "$BACKUP_DIR/"
    log "✅ Data backup created"
fi

if [ -f "./docker-compose.yml" ]; then
    cp ./docker-compose.yml "$BACKUP_DIR/"
    log "✅ Configuration backup created"
fi

# Step 2: Pull latest images
log "📥 Pulling latest Docker images..."
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f docker-compose.yml pull
else
    docker-compose -f docker-compose.dev.yml pull
fi

# Step 3: Build application image
log "🔨 Building application image..."
docker build -t ai-agent:latest .

# Step 4: Stop existing services
log "🛑 Stopping existing services..."
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f docker-compose.yml down || true
else
    docker-compose -f docker-compose.dev.yml down || true
fi

# Step 5: Start new services
log "🚀 Starting new services..."
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f docker-compose.yml up -d
else
    docker-compose -f docker-compose.dev.yml up -d
fi

# Step 6: Wait for services to start
log "⏳ Waiting for services to start..."
sleep 30

# Step 7: Run health checks
log "🏥 Running health checks..."
python3 scripts/production_validator.py --validate-only

# Step 8: Run smoke tests
log "🧪 Running smoke tests..."
python3 -m pytest tests/test_production_api.py::TestProductionAPI::test_health_check -v

log "✅ Deployment completed successfully!"
log "📊 View logs: tail -f $LOG_FILE"
log "📈 Monitor metrics: http://localhost:3000 (Grafana)"
log "🔍 API docs: http://localhost:8000/docs"