#!/bin/bash

#
# Manage the deployed user research application
# Usage: ./aws-automation/manage_user_research.sh [command]
#

# Load shared library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aws-lib.sh"

PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REMOTE_DIR="/home/ubuntu/persona-red-teaming"

# Colors
BOLD='\033[1m'
NC='\033[0m'

# ==================== HELPER FUNCTIONS ====================

show_usage() {
    cat << EOF
${BOLD}Usage:${NC}
  ./aws-automation/manage_user_research.sh [command]

${BOLD}Commands:${NC}
  status          Show deployment status and URLs
  logs            View application logs
  restart         Restart the application
  stop            Stop the application (keeps instance running)
  start           Start the application
  ssh             SSH into the instance
  backup-db       Backup database to local machine
  update          Update application code and restart
  instance-stop   Stop EC2 instance (saves costs)
  instance-start  Start EC2 instance
  terminate       Terminate EC2 instance (WARNING: deletes everything)
  help            Show this help message

${BOLD}Examples:${NC}
  ./aws-automation/manage_user_research.sh status
  ./aws-automation/manage_user_research.sh logs
  ./aws-automation/manage_user_research.sh backup-db
EOF
}

# Find the most recent user research deployment
find_deployment() {
    if [ ! -f "$PROJECT_DIR/.aws-experiments.json" ]; then
        lib_error "No deployments found. Deploy first with: ./aws-automation/deploy_user_research.sh"
        exit 1
    fi

    # Find the most recent research deployment
    DEPLOYMENT=$(jq -r '.experiments[] | select(.experiment_name == "user-research-app") | select(.status != "terminated") | .id' "$PROJECT_DIR/.aws-experiments.json" | tail -1)

    if [ -z "$DEPLOYMENT" ]; then
        lib_error "No active deployment found"
        exit 1
    fi

    # Load deployment metadata
    INSTANCE_ID=$(jq -r --arg id "$DEPLOYMENT" '.experiments[] | select(.id == $id) | .instance_id' "$PROJECT_DIR/.aws-experiments.json")
    PUBLIC_IP=$(jq -r --arg id "$DEPLOYMENT" '.experiments[] | select(.id == $id) | .public_ip' "$PROJECT_DIR/.aws-experiments.json")

    lib_debug "Found deployment: $DEPLOYMENT"
    lib_debug "Instance: $INSTANCE_ID"
    lib_debug "IP: $PUBLIC_IP"
}

# Get current instance state from AWS
get_instance_state() {
    local state=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null || echo "not-found")
    echo "$state"
}

# Get current public IP from AWS (in case it changed after stop/start)
refresh_public_ip() {
    local new_ip=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null || echo "")

    if [ -n "$new_ip" ] && [ "$new_ip" != "$PUBLIC_IP" ]; then
        lib_warn "Public IP changed from $PUBLIC_IP to $new_ip"
        PUBLIC_IP="$new_ip"
        update_experiment_ip "$DEPLOYMENT" "$new_ip"
    fi
}

# ==================== COMMANDS ====================

cmd_status() {
    find_deployment
    refresh_public_ip

    local instance_state=$(get_instance_state)

    lib_log "=========================================="
    lib_log "Deployment Status"
    lib_log "=========================================="
    echo ""
    echo "Deployment ID: $DEPLOYMENT"
    echo "Instance ID: $INSTANCE_ID"
    echo "Instance State: $instance_state"
    echo "Public IP: $PUBLIC_IP"
    echo ""

    if [ "$instance_state" != "running" ]; then
        lib_warn "Instance is not running (state: $instance_state)"
        echo ""
        echo "To start the instance:"
        echo "  ./aws-automation/manage_user_research.sh instance-start"
        return
    fi

    # Check application health
    echo "Application Status:"
    echo -n "  Backend: "
    if curl -s -f "http://$PUBLIC_IP:8000/health" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
    else
        echo -e "${RED}✗ Not responding${NC}"
    fi

    echo -n "  Frontend: "
    if curl -s -f "http://$PUBLIC_IP" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
    else
        echo -e "${RED}✗ Not responding${NC}"
    fi

    echo ""
    echo "URLs:"
    echo "  Frontend: http://$PUBLIC_IP"
    echo "  Backend: http://$PUBLIC_IP:8000"
    echo "  Health: http://$PUBLIC_IP:8000/health"
    echo ""
    echo "SSH:"
    echo "  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP"
    echo ""
}

cmd_logs() {
    find_deployment
    refresh_public_ip

    local instance_state=$(get_instance_state)
    if [ "$instance_state" != "running" ]; then
        lib_error "Instance is not running. Start it first."
        exit 1
    fi

    lib_log "Connecting to view logs (Ctrl+C to exit)..."
    echo ""

    ssh -i "$KEY_PATH" -t ubuntu@"$PUBLIC_IP" "cd $REMOTE_DIR && docker-compose logs -f"
}

cmd_restart() {
    find_deployment
    refresh_public_ip

    local instance_state=$(get_instance_state)
    if [ "$instance_state" != "running" ]; then
        lib_error "Instance is not running. Start it first."
        exit 1
    fi

    lib_log "Restarting application..."

    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" << ENDSSH
        cd $REMOTE_DIR
        docker-compose restart
        echo "✓ Application restarted"
ENDSSH

    lib_log "✓ Application restarted"
    sleep 5
    cmd_status
}

cmd_stop() {
    find_deployment
    refresh_public_ip

    local instance_state=$(get_instance_state)
    if [ "$instance_state" != "running" ]; then
        lib_warn "Instance is already stopped"
        return
    fi

    lib_log "Stopping application containers..."

    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" << ENDSSH
        cd $REMOTE_DIR
        docker-compose down
        echo "✓ Application stopped"
ENDSSH

    lib_log "✓ Application stopped (instance still running)"
}

cmd_start() {
    find_deployment
    refresh_public_ip

    local instance_state=$(get_instance_state)
    if [ "$instance_state" != "running" ]; then
        lib_error "Instance is not running. Start it with: ./aws-automation/manage_user_research.sh instance-start"
        exit 1
    fi

    lib_log "Starting application containers..."

    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" << ENDSSH
        cd $REMOTE_DIR
        docker-compose up -d
        echo "✓ Application started"
ENDSSH

    lib_log "✓ Application started"
    sleep 5
    cmd_status
}

cmd_ssh() {
    find_deployment
    refresh_public_ip

    local instance_state=$(get_instance_state)
    if [ "$instance_state" != "running" ]; then
        lib_error "Instance is not running. Start it first."
        exit 1
    fi

    lib_log "Connecting to instance..."
    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP"
}

cmd_backup_db() {
    find_deployment
    refresh_public_ip

    local instance_state=$(get_instance_state)
    if [ "$instance_state" != "running" ]; then
        lib_error "Instance is not running. Start it first."
        exit 1
    fi

    local backup_dir="$PROJECT_DIR/database-backups"
    mkdir -p "$backup_dir"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$backup_dir/history_$timestamp.db"

    lib_log "Backing up database..."

    scp -i "$KEY_PATH" \
        ubuntu@"$PUBLIC_IP":"$REMOTE_DIR/data/history.db" \
        "$backup_file" 2>/dev/null || {
            lib_warn "Database file not found or empty. Application may not have been used yet."
            return
        }

    lib_log "✓ Database backed up to: $backup_file"

    # Show database stats
    if command -v sqlite3 >/dev/null 2>&1; then
        echo ""
        lib_log "Database Statistics:"
        sqlite3 "$backup_file" << 'EOF'
.mode column
.headers on
SELECT 'Sessions' as table_name, COUNT(*) as count FROM sessions
UNION ALL
SELECT 'Personas', COUNT(*) FROM personas
UNION ALL
SELECT 'Prompts', COUNT(*) FROM prompts
UNION ALL
SELECT 'Guard Results', COUNT(*) FROM guard_results
UNION ALL
SELECT 'User Feedback', COUNT(*) FROM user_feedback;
EOF
    fi
}

cmd_update() {
    find_deployment
    refresh_public_ip

    local instance_state=$(get_instance_state)
    if [ "$instance_state" != "running" ]; then
        lib_error "Instance is not running. Start it first."
        exit 1
    fi

    lib_log "Updating application code..."

    # Upload new code
    rsync -avz --progress \
        -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no" \
        --exclude 'venv/' \
        --exclude 'logs-*/' \
        --exclude '.git/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.DS_Store' \
        --exclude 'node_modules/' \
        --exclude 'data/' \
        --exclude '*.db' \
        --exclude 'aws-results-*/' \
        "$PROJECT_DIR/" ubuntu@"$PUBLIC_IP":"$REMOTE_DIR/"

    lib_log "Rebuilding and restarting application..."

    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" << ENDSSH
        cd $REMOTE_DIR
        docker-compose down
        docker-compose up -d --build
        echo "✓ Application updated and restarted"
ENDSSH

    lib_log "✓ Update complete"
    sleep 10
    cmd_status
}

cmd_instance_stop() {
    find_deployment

    local instance_state=$(get_instance_state)
    if [ "$instance_state" == "stopped" ]; then
        lib_warn "Instance is already stopped"
        return
    fi

    lib_warn "This will stop the EC2 instance (application will be unavailable)"
    read -p "Continue? (y/N): " confirm

    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        lib_log "Cancelled"
        return
    fi

    lib_log "Stopping EC2 instance..."

    aws ec2 stop-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --output text >/dev/null

    lib_log "✓ Instance stopping (this takes ~1 minute)"
    lib_log "Instance will stop incurring compute charges"
    lib_log "You'll still pay for EBS storage (~\$3/month for 30GB)"
}

cmd_instance_start() {
    find_deployment

    local instance_state=$(get_instance_state)
    if [ "$instance_state" == "running" ]; then
        lib_warn "Instance is already running"
        cmd_status
        return
    fi

    lib_log "Starting EC2 instance..."

    aws ec2 start-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --output text >/dev/null

    lib_log "Waiting for instance to be running..."
    aws ec2 wait instance-running \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID"

    # Get new public IP
    refresh_public_ip

    lib_log "✓ Instance started"
    lib_log "New public IP: $PUBLIC_IP"
    lib_log ""
    lib_warn "IMPORTANT: The public IP has changed!"
    lib_warn "You need to update your frontend to use the new API URL:"
    lib_warn "  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP"
    lib_warn "  cd ~/persona-red-teaming"
    lib_warn "  sed -i 's|VITE_API_BASE_URL=.*|VITE_API_BASE_URL=http://$PUBLIC_IP:8000|' docker-compose.yml"
    lib_warn "  docker-compose up -d --build"
    echo ""

    sleep 5
    cmd_status
}

cmd_terminate() {
    find_deployment

    lib_error "=========================================="
    lib_error "WARNING: This will permanently delete:"
    lib_error "  - The EC2 instance"
    lib_error "  - All application data"
    lib_error "  - The database"
    lib_error "=========================================="
    echo ""
    read -p "Type 'DELETE' to confirm: " confirm

    if [ "$confirm" != "DELETE" ]; then
        lib_log "Cancelled"
        return
    fi

    # Backup database first
    lib_log "Creating final database backup..."
    cmd_backup_db || lib_warn "Could not backup database"

    lib_log "Terminating EC2 instance..."

    aws ec2 terminate-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --output text >/dev/null

    # Update metadata
    update_experiment_status "$DEPLOYMENT" "terminated"

    lib_log "✓ Instance terminated: $INSTANCE_ID"
    lib_log "Deployment removed from tracking"
}

# ==================== MAIN ====================

main() {
    local command="${1:-status}"

    case "$command" in
        status)
            cmd_status
            ;;
        logs)
            cmd_logs
            ;;
        restart)
            cmd_restart
            ;;
        stop)
            cmd_stop
            ;;
        start)
            cmd_start
            ;;
        ssh)
            cmd_ssh
            ;;
        backup-db)
            cmd_backup_db
            ;;
        update)
            cmd_update
            ;;
        instance-stop)
            cmd_instance_stop
            ;;
        instance-start)
            cmd_instance_start
            ;;
        terminate)
            cmd_terminate
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            lib_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
