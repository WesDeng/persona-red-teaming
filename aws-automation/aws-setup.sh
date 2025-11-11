#!/bin/bash

#
# One-time setup script for AWS prerequisites
# Run this once before using deploy_and_run.sh
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log "=========================================="
log "AWS Prerequisites Setup"
log "=========================================="
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    log "Installing AWS CLI..."
    brew install awscli
else
    log "✓ AWS CLI already installed"
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    log "Installing jq..."
    brew install jq
else
    log "✓ jq already installed"
fi

# Configure AWS credentials
log "Configuring AWS credentials..."
warn "You'll need your AWS Access Key ID and Secret Access Key"
warn "Get these from: https://console.aws.amazon.com/iam/home#/security_credentials"
echo ""

aws configure

# Get AWS region from config
AWS_REGION=$(aws configure get region)
log "Using AWS region: $AWS_REGION"

# Verify AWS credentials
log "Verifying AWS credentials..."
if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
    log "✓ AWS credentials verified"
    log "  Account ID: $ACCOUNT_ID"
    log "  User/Role: $USER_ARN"
else
    echo ""
    warn "⚠️  AWS credential verification failed!"
    warn "Please check your Access Key and Secret Key are correct"
    exit 1
fi

# Verify EC2 access
log "Verifying EC2 access..."
if aws ec2 describe-regions --region "$AWS_REGION" &> /dev/null; then
    log "✓ EC2 access verified"
else
    echo ""
    warn "⚠️  Cannot access EC2 service!"
    warn "Please ensure your IAM user has EC2 permissions"
    exit 1
fi

# Check if .env file exists
log "Checking for .env file..."
# This script is in aws-automation/, so project root is parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$PROJECT_DIR/.env" ]; then
    log "✓ .env file found"

    # Check if it has the required API keys
    if grep -q "OPENAI_API_KEY" "$PROJECT_DIR/.env" || grep -q "TOGETHER_API_KEY" "$PROJECT_DIR/.env"; then
        log "✓ API keys found in .env"
    else
        warn "⚠️  .env file exists but no API keys found"
        warn "Make sure to add OPENAI_API_KEY or TOGETHER_API_KEY"
    fi
else
    warn "⚠️  .env file not found at: $PROJECT_DIR/.env"
    warn "You'll need to create it with your API keys before running experiments"
    warn "Example:"
    warn "  OPENAI_API_KEY=sk-..."
    warn "  TOGETHER_API_KEY=..."
fi

echo ""

# Create EC2 key pair
KEY_NAME="persona-redteaming-key"
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"

if [ -f "$KEY_PATH" ]; then
    warn "Key pair already exists at: $KEY_PATH"
else
    log "Creating EC2 key pair..."

    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text > "$KEY_PATH"

    chmod 400 "$KEY_PATH"

    log "✓ Key pair created: $KEY_PATH"
fi

echo ""
log "=========================================="
log "✓ Setup Complete!"
log "=========================================="
echo ""
log "Next steps:"
log ""
log "1. Deploy and run an experiment:"
log "   ./aws-automation/deploy_and_run.sh configs/qwen-target.yml"
log ""
log "2. Check status of running experiments:"
log "   ./aws-automation/check_experiments.sh"
log ""
warn "Note: Default region is us-east-2. Edit aws-automation/deploy_and_run.sh to change."
