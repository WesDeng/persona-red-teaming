#!/bin/bash

#
# Deploy the persona red-teaming user research application on AWS EC2
# This deploys the frontend + backend web application for data collection
# Usage: ./aws-automation/deploy_user_research.sh
#

set -e  # Exit on error

# Load shared library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aws-lib.sh"

# ==================== CONFIGURATION ====================

# AWS Configuration
AWS_REGION="${AWS_REGION:-us-east-2}"  # Change to your preferred region
INSTANCE_TYPE="t3.small"  # Enough for Docker web app (t2.micro too small)
KEY_NAME="${KEY_NAME:-persona-redteaming-key}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/${KEY_NAME}.pem}"
SECURITY_GROUP="persona-research-app-sg"
SPOT_INSTANCE=false  # Use on-demand for persistent web app

# Project Configuration
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REMOTE_DIR="/home/ubuntu/persona-red-teaming"
DEPLOYMENT_NAME="user-research-app"
DEPLOYMENT_ID="research-$(date +%Y%m%d-%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== HELPER FUNCTIONS ====================

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${BLUE}[SUCCESS]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    log "Checking requirements..."

    command -v aws >/dev/null 2>&1 || error "AWS CLI not found. Install with: brew install awscli"
    command -v jq >/dev/null 2>&1 || error "jq not found. Install with: brew install jq"

    if [ ! -f "$KEY_PATH" ]; then
        error "SSH key not found at $KEY_PATH. Run aws-setup.sh first or create an EC2 key pair."
    fi

    if [ ! -f "$PROJECT_DIR/.env" ]; then
        warn ".env file not found at $PROJECT_DIR/.env"
        warn "You'll need to configure API keys after deployment"
    fi

    log "✓ All requirements met"
}

# Create security group for web application
setup_security_group() {
    log "Setting up security group for web application..."

    # Check if security group exists
    SG_ID=$(aws ec2 describe-security-groups \
        --region "$AWS_REGION" \
        --filters "Name=group-name,Values=$SECURITY_GROUP" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null || echo "None")

    # Get current IP
    MY_IP=$(curl -s https://checkip.amazonaws.com)

    if [ "$SG_ID" = "None" ]; then
        log "Creating security group..."
        SG_ID=$(aws ec2 create-security-group \
            --region "$AWS_REGION" \
            --group-name "$SECURITY_GROUP" \
            --description "Security group for persona red-teaming research application" \
            --query 'GroupId' \
            --output text)

        log "✓ Security group created: $SG_ID"

        # Add rules for web application
        log "Adding security group rules..."

        # SSH from current IP
        aws ec2 authorize-security-group-ingress \
            --region "$AWS_REGION" \
            --group-id "$SG_ID" \
            --protocol tcp \
            --port 22 \
            --cidr "${MY_IP}/32" || true

        # HTTP (port 80) from anywhere
        aws ec2 authorize-security-group-ingress \
            --region "$AWS_REGION" \
            --group-id "$SG_ID" \
            --protocol tcp \
            --port 80 \
            --cidr 0.0.0.0/0 || true

        # Backend API (port 8000) from anywhere
        aws ec2 authorize-security-group-ingress \
            --region "$AWS_REGION" \
            --group-id "$SG_ID" \
            --protocol tcp \
            --port 8000 \
            --cidr 0.0.0.0/0 || true

        log "✓ Security group rules configured"
    else
        log "✓ Security group already exists: $SG_ID"

        # Ensure current IP has SSH access
        log "Ensuring current IP ($MY_IP) has SSH access..."
        aws ec2 authorize-security-group-ingress \
            --region "$AWS_REGION" \
            --group-id "$SG_ID" \
            --protocol tcp \
            --port 22 \
            --cidr "${MY_IP}/32" 2>/dev/null || log "IP already authorized"
    fi
}

# Get the correct AMI ID for the region
get_ami_id() {
    log "Finding Ubuntu 22.04 AMI in region $AWS_REGION..."

    AMI_ID=$(aws ec2 describe-images \
        --region "$AWS_REGION" \
        --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text)

    log "✓ Using AMI: $AMI_ID"
}

# Launch EC2 instance
launch_instance() {
    log "Launching EC2 instance ($INSTANCE_TYPE)..."

    # Launch on-demand instance (not spot - we want it to stay up)
    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$AWS_REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$DEPLOYMENT_NAME}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    log "✓ Instance launched: $INSTANCE_ID"

    # Wait for instance to be running
    log "Waiting for instance to be running..."
    aws ec2 wait instance-running \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID"

    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    log "✓ Instance running at: $PUBLIC_IP"

    # Wait for SSH to be ready
    log "Waiting for SSH to be ready..."
    for i in {1..30}; do
        if ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@"$PUBLIC_IP" "echo 'SSH ready'" &>/dev/null; then
            log "✓ SSH is ready"
            break
        fi
        echo -n "."
        sleep 10
    done
    echo ""
}

# Setup instance with Docker
setup_instance() {
    log "Setting up instance with Docker..."

    ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" << 'ENDSSH'
        set -e

        # Update system
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
            docker.io \
            docker-compose \
            git \
            curl \
            htop

        # Start Docker
        sudo systemctl start docker
        sudo systemctl enable docker

        # Add ubuntu user to docker group
        sudo usermod -aG docker ubuntu

        echo "✓ Docker setup complete"
ENDSSH

    log "✓ Instance setup complete"

    # Need to reconnect for docker group to take effect
    log "Reconnecting to apply docker group..."
    sleep 2
}

# Upload code
upload_code() {
    log "Uploading application code..."

    # Create remote directory
    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" "mkdir -p $REMOTE_DIR"

    # Upload project files
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

    log "✓ Code uploaded"
}

# Configure environment variables
configure_env() {
    log "Configuring environment variables..."

    # Check if .env exists locally
    if [ -f "$PROJECT_DIR/.env" ]; then
        log "Uploading .env file..."
        scp -i "$KEY_PATH" "$PROJECT_DIR/.env" ubuntu@"$PUBLIC_IP":"$REMOTE_DIR/.env"
    else
        warn "No .env file found locally. Creating template..."

        ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" << 'ENDSSH'
            cat > /home/ubuntu/persona-red-teaming/.env << 'EOF'
# OpenAI API Key (required)
OPENAI_API_KEY=your-key-here

# Google AI API Key (optional, for Gemini models)
GOOGLE_API_KEY=your-key-here

# Database path
DATABASE_PATH=/app/data/history.db
EOF
            echo "✓ Created .env template"
ENDSSH

        warn "IMPORTANT: You need to edit .env on the server with your API keys!"
        warn "Run: ssh -i $KEY_PATH ubuntu@$PUBLIC_IP"
        warn "Then: nano ~/persona-red-teaming/.env"
    fi

    # Update docker-compose.yml with public IP
    log "Updating docker-compose.yml with public IP..."

    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" << ENDSSH
        cd $REMOTE_DIR

        # Update VITE_API_BASE_URL to use public IP
        sed -i "s|VITE_API_BASE_URL=http://localhost:8000|VITE_API_BASE_URL=http://$PUBLIC_IP:8000|g" docker-compose.yml

        echo "✓ Updated docker-compose.yml"
ENDSSH

    log "✓ Environment configured"
}

# Build and start application
start_application() {
    log "Building and starting Docker containers..."
    log "This will take 5-10 minutes on first build..."

    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" << ENDSSH
        set -e
        cd $REMOTE_DIR

        # Create data directory
        mkdir -p data

        # Build and start containers
        docker-compose up -d --build

        echo "✓ Containers started"
ENDSSH

    log "✓ Application started"
}

# Wait for application to be healthy
wait_for_health() {
    log "Waiting for application to be healthy..."

    for i in {1..30}; do
        # Check backend health
        if curl -s -f "http://$PUBLIC_IP:8000/health" >/dev/null 2>&1; then
            log "✓ Backend is healthy"

            # Check frontend
            if curl -s -f "http://$PUBLIC_IP" >/dev/null 2>&1; then
                log "✓ Frontend is healthy"
                return 0
            fi
        fi
        echo -n "."
        sleep 10
    done

    warn "Health check timeout, but containers may still be starting..."
    warn "Check manually: curl http://$PUBLIC_IP:8000/health"
}

# Show deployment summary
show_summary() {
    log ""
    log "=========================================="
    success "DEPLOYMENT COMPLETE!"
    log "=========================================="
    log ""
    log "Instance Information:"
    log "  Instance ID: $INSTANCE_ID"
    log "  Public IP: $PUBLIC_IP"
    log "  Region: $AWS_REGION"
    log ""
    log "Application URLs:"
    success "  Frontend: http://$PUBLIC_IP"
    success "  Backend API: http://$PUBLIC_IP:8000"
    success "  Health Check: http://$PUBLIC_IP:8000/health"
    log ""
    log "SSH Access:"
    log "  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP"
    log ""
    log "Database Access:"
    log "  Location: ~/persona-red-teaming/data/history.db"
    log "  Query: sqlite3 ~/persona-red-teaming/data/history.db"
    log ""
    log "Management Commands:"
    log "  View logs: docker-compose logs -f"
    log "  Restart: docker-compose restart"
    log "  Stop: docker-compose down"
    log "  Start: docker-compose up -d"
    log ""
    log "Database Backup:"
    log "  scp -i $KEY_PATH ubuntu@$PUBLIC_IP:~/persona-red-teaming/data/history.db ./backup.db"
    log ""

    if [ ! -f "$PROJECT_DIR/.env" ] || ! grep -q "OPENAI_API_KEY=sk-" "$PROJECT_DIR/.env" 2>/dev/null; then
        warn "=========================================="
        warn "ACTION REQUIRED: Configure API Keys"
        warn "=========================================="
        warn "1. SSH into server: ssh -i $KEY_PATH ubuntu@$PUBLIC_IP"
        warn "2. Edit .env file: nano ~/persona-red-teaming/.env"
        warn "3. Add your OpenAI API key"
        warn "4. Restart application: cd ~/persona-red-teaming && docker-compose restart"
        warn ""
    fi

    log "Cost Estimate: ~\$0.02/hour (~\$15/month for t3.small)"
    log ""
    log "To stop instance and save costs:"
    log "  aws ec2 stop-instances --region $AWS_REGION --instance-ids $INSTANCE_ID"
    log ""
    log "To terminate instance completely:"
    log "  aws ec2 terminate-instances --region $AWS_REGION --instance-ids $INSTANCE_ID"
    log ""
    log "=========================================="
}

# Save deployment metadata
save_deployment_metadata() {
    log "Saving deployment metadata..."

    # Use the shared library function
    save_experiment_metadata \
        "$DEPLOYMENT_ID" \
        "$INSTANCE_ID" \
        "user-research-app" \
        "$DEPLOYMENT_NAME" \
        "$PUBLIC_IP" \
        "running"

    # Also create a simple deployment info file
    cat > "$PROJECT_DIR/.last-deployment.txt" << EOF
Deployment ID: $DEPLOYMENT_ID
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
Region: $AWS_REGION
Deployed: $(date)

Frontend: http://$PUBLIC_IP
Backend: http://$PUBLIC_IP:8000

SSH: ssh -i $KEY_PATH ubuntu@$PUBLIC_IP
EOF

    log "✓ Metadata saved to .aws-experiments.json and .last-deployment.txt"
}

# Cleanup on error
cleanup_on_error() {
    error "Deployment failed. Cleaning up..."

    if [ -n "$INSTANCE_ID" ]; then
        warn "Terminating instance: $INSTANCE_ID"
        aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" || true
    fi
}

# ==================== MAIN EXECUTION ====================

main() {
    log "=========================================="
    log "AWS EC2 User Research App Deployment"
    log "=========================================="
    log "This will deploy the persona red-teaming"
    log "web application for user research and"
    log "data collection."
    log ""

    # Trap errors
    trap cleanup_on_error ERR

    # Pre-flight checks
    check_requirements

    # Setup AWS resources
    setup_security_group
    get_ami_id

    # Launch and configure instance
    launch_instance
    setup_instance

    # Deploy application
    upload_code
    configure_env
    start_application

    # Wait for health
    wait_for_health

    # Save metadata
    save_deployment_metadata

    # Show summary
    show_summary
}

# Run main function
main "$@"
