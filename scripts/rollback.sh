#!/bin/bash
# scripts/rollback.sh - Rollback script

set -euo pipefail

BACKUP_DIR=${1:-$(ls -t ./backups/ | head -n1)}
LOG_FILE="./logs/rollback_$(date +%Y%m%d_%H%M%S).log"

echo "🔄 AI Agent Rollback Script"
echo "Backup Directory: ./backups/$BACKUP_DIR"
echo "Log File: $LOG_FILE"
echo "================================="

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

handle_error() {
    log "❌ Error on line $1"
    log "Rollback failed. Manual intervention required."
    exit 1
}

trap 'handle_error $LINENO' ERR

# Step 1: Stop current services
log "🛑 Stopping current services..."
docker-compose down

# Step 2: Restore data
log "📦 Restoring data from backup..."
if [ -d "./backups/$BACKUP_DIR/data" ]; then
    rm -rf ./data
    cp -r "./backups/$BACKUP_DIR/data" ./
    log "✅ Data restored"
fi

# Step 3: Restore configuration
if [ -f "./backups/$BACKUP_DIR/docker-compose.yml" ]; then
    cp "./backups/$BACKUP_DIR/docker-compose.yml" ./
    log "✅ Configuration restored"
fi

# Step 4: Start services with restored configuration
log "🚀 Starting services with restored configuration..."
docker-compose up -d

# Step 5: Wait and verify
log "⏳ Waiting for services to start..."
sleep 30

log "🏥 Running health check..."
python3 scripts/production_validator.py --validate-only

log "✅ Rollback completed successfully!"