#!/bin/bash

#
# Full automation script for running persona red-teaming experiments on AWS EC2
# Usage: ./deploy_and_run.sh <config-file-1> [config-file-2] [config-file-3] ...
# Example: ./deploy_and_run.sh qwen-target.yml gpt4-target.yml
#

set -e  # Exit on error

# Load shared library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aws-lib.sh"

# ==================== CONFIGURATION ====================

# AWS Configuration
AWS_REGION="us-east-2"  # Change to your preferred region
INSTANCE_TYPE="t3.medium"  # Cheap and sufficient for API-based workloads
AMI_ID="ami-0c55b159cbfafe1f0"  # Ubuntu 22.04 LTS (update for your region)
KEY_NAME="persona-redteaming-key"  # Your EC2 key pair name
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"  # Local path to your private key
SECURITY_GROUP="persona-redteaming-sg"  # Security group name
SPOT_INSTANCE=false  # Use spot instances for ~70% cost savings

# Project Configuration
# This script is in aws-automation/, so project root is parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REMOTE_DIR="/home/ubuntu/persona-red-teaming"

# Multiple config files support
if [ $# -eq 0 ]; then
    CONFIG_FILES=("configs/qwen-target.yml")  # Default
else
    CONFIG_FILES=("$@")
fi

# Single instance mode - reuse instance for multiple experiments
REUSE_INSTANCE=true  # Set to true to run all experiments on one instance

# These will be set for each experiment in the loop
CONFIG_FILE=""
CONFIG_FILE_FULL_PATH=""
EXPERIMENT_NAME=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if required tools are installed
check_requirements() {
    log "Checking requirements..."

    command -v aws >/dev/null 2>&1 || error "AWS CLI not found. Install with: brew install awscli"
    command -v jq >/dev/null 2>&1 || error "jq not found. Install with: brew install jq"

    if [ ! -f "$KEY_PATH" ]; then
        error "SSH key not found at $KEY_PATH. Please create an EC2 key pair first."
    fi

    if [ ! -f "$PROJECT_DIR/.env" ]; then
        error ".env file not found at $PROJECT_DIR/.env. Please create it with your API keys."
    fi

    log "✓ All requirements met"
}

# Check if a specific config file exists
check_config_file() {
    local config_file="$1"
    if [ ! -f "$config_file" ]; then
        error "Config file not found: $config_file"
    fi
}

# Create security group if it doesn't exist
setup_security_group() {
    log "Setting up security group..."

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
            --description "Security group for persona red-teaming experiments" \
            --query 'GroupId' \
            --output text)

        log "✓ Security group created: $SG_ID"
    else
        log "✓ Security group already exists: $SG_ID"
    fi

    # Always ensure current IP has SSH access (idempotent - won't fail if already exists)
    log "Ensuring current IP ($MY_IP) has SSH access..."
    aws ec2 authorize-security-group-ingress \
        --region "$AWS_REGION" \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr "${MY_IP}/32" 2>/dev/null || log "IP already authorized (or failed to add)"
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

    if [ "$SPOT_INSTANCE" = true ]; then
        log "Using spot instance for cost savings..."

        # Create launch specification (without TagSpecifications - not supported in spot)
        LAUNCH_SPEC=$(cat <<EOF
{
    "ImageId": "$AMI_ID",
    "InstanceType": "$INSTANCE_TYPE",
    "KeyName": "$KEY_NAME",
    "SecurityGroupIds": ["$SG_ID"],
    "BlockDeviceMappings": [{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 50,
            "VolumeType": "gp3"
        }
    }]
}
EOF
)

        # Request spot instance
        SPOT_REQUEST_ID=$(aws ec2 request-spot-instances \
            --region "$AWS_REGION" \
            --spot-price "0.10" \
            --instance-count 1 \
            --type "one-time" \
            --launch-specification "$LAUNCH_SPEC" \
            --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
            --output text)

        log "Waiting for spot request to be fulfilled..."
        aws ec2 wait spot-instance-request-fulfilled \
            --region "$AWS_REGION" \
            --spot-instance-request-ids "$SPOT_REQUEST_ID"

        INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
            --region "$AWS_REGION" \
            --spot-instance-request-ids "$SPOT_REQUEST_ID" \
            --query 'SpotInstanceRequests[0].InstanceId' \
            --output text)

        # Tag the instance after creation
        aws ec2 create-tags \
            --region "$AWS_REGION" \
            --resources "$INSTANCE_ID" \
            --tags "Key=Name,Value=persona-redteaming-$EXPERIMENT_NAME"
    else
        # Launch regular on-demand instance
        INSTANCE_ID=$(aws ec2 run-instances \
            --region "$AWS_REGION" \
            --image-id "$AMI_ID" \
            --instance-type "$INSTANCE_TYPE" \
            --key-name "$KEY_NAME" \
            --security-group-ids "$SG_ID" \
            --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3}" \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=persona-redteaming-$EXPERIMENT_NAME}]" \
            --query 'Instances[0].InstanceId' \
            --output text)
    fi

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

# Setup instance
setup_instance() {
    log "Setting up instance..."

    ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" << 'ENDSSH'
        set -e

        # Clean apt lists to fix potential GPG errors
        sudo rm -rf /var/lib/apt/lists/*

        # Update system
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
            python3-pip python3.10-venv git tmux htop

        echo "Setup complete"
ENDSSH

    log "✓ Instance setup complete"
}

# Upload code
upload_code() {
    log "Uploading code to instance..."

    # Create remote directory
    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" "mkdir -p $REMOTE_DIR"

    # Upload entire project directory (excluding large files)
    rsync -avz --progress \
        -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no" \
        --exclude 'venv/' \
        --exclude 'logs-*/' \
        --exclude '.git/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.DS_Store' \
        "$PROJECT_DIR/" ubuntu@"$PUBLIC_IP":"$REMOTE_DIR/"

    log "✓ Code uploaded"
}

# Install dependencies and setup environment
install_dependencies() {
    log "Installing Python dependencies..."

    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" << ENDSSH
        set -e
        cd $REMOTE_DIR

        # Create virtual environment
        python3 -m venv venv
        source venv/bin/activate

        # Upgrade pip
        pip install --upgrade pip

        # Install package
        pip install -e .

        # Download NLTK data
        python download_nltk_data.py

        echo "Dependencies installed"
ENDSSH

    log "✓ Dependencies installed"
}

# Run experiment
run_experiment() {
    log "Starting experiment: $CONFIG_FILE"

    # Create a script to run on the remote machine
    ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" << ENDSSH
        set -e
        cd $REMOTE_DIR
        source venv/bin/activate

        # Run experiment in tmux and capture exit code
        tmux new-session -d -s redteam "python -m rainbowplus.rainbowplus --config_file $CONFIG_FILE 2>&1 | tee experiment.log; echo \\\$? > experiment_exit_code.txt"

        echo "Experiment started in tmux session 'redteam'"
ENDSSH

    log "✓ Experiment started"
}

# Monitor experiment progress
monitor_experiment() {
    log "Monitoring experiment progress..."
    log "This may take several hours depending on your configuration..."

    while true; do
        # Check if tmux session is still running (with timeout and retry)
        SESSION_RUNNING=$(ssh -i "$KEY_PATH" \
            -o ConnectTimeout=30 \
            -o ServerAliveInterval=60 \
            -o ServerAliveCountMax=3 \
            ubuntu@"$PUBLIC_IP" \
            "tmux list-sessions 2>/dev/null | grep redteam || echo 'none'" 2>/dev/null || echo 'ssh_error')

        # Handle SSH connection errors
        if [ "$SESSION_RUNNING" = "ssh_error" ]; then
            warn "SSH connection failed, retrying in 30 seconds..."
            sleep 30
            continue
        fi

        if [ "$SESSION_RUNNING" = "none" ]; then
            log "✓ Experiment completed!"
            break
        fi

        # Show last few lines of log (with timeout, don't fail if this errors)
        echo ""
        log "Recent log output:"
        ssh -i "$KEY_PATH" \
            -o ConnectTimeout=30 \
            -o ServerAliveInterval=60 \
            -o ServerAliveCountMax=3 \
            ubuntu@"$PUBLIC_IP" \
            "tail -n 5 $REMOTE_DIR/experiment.log 2>/dev/null || echo 'Log not available yet'" 2>/dev/null || warn "Could not fetch logs (network issue)"

        # Wait before checking again
        sleep 60
    done
}

# Run analysis scripts
run_analysis() {
    log "Running analysis scripts..."

    # Determine the log directory based on config (with error handling)
    # Make LOG_DIR global so download_results can use it
    LOG_DIR=$(ssh -i "$KEY_PATH" \
        -o ConnectTimeout=30 \
        -o ServerAliveInterval=60 \
        ubuntu@"$PUBLIC_IP" \
        "cd $REMOTE_DIR && source venv/bin/activate && python -c \"
import yaml
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)
print(config.get('log_dir', './logs'))
\"" 2>/dev/null || echo "logs")

    # Find the actual log directory (should be logs-*/model-name/harmbench)
    ACTUAL_LOG_DIR=$(ssh -i "$KEY_PATH" \
        -o ConnectTimeout=30 \
        ubuntu@"$PUBLIC_IP" \
        "find $REMOTE_DIR/$LOG_DIR -type d -name 'harmbench' | head -1" 2>/dev/null)

    if [ -z "$ACTUAL_LOG_DIR" ]; then
        warn "Could not find harmbench log directory, skipping analysis"
        return 0
    fi

    log "Log directory: $ACTUAL_LOG_DIR"

    # Run both analysis scripts (don't fail the whole script if analysis fails)
    ssh -i "$KEY_PATH" \
        -o ConnectTimeout=30 \
        -o ServerAliveInterval=60 \
        ubuntu@"$PUBLIC_IP" << ENDSSH || warn "Analysis failed, but continuing..."
        set -e
        cd $REMOTE_DIR
        source venv/bin/activate

        # Analysis 1: analyze_comprehensive_logs.py
        echo "Running comprehensive log analysis..."
        python analyze_comprehensive_logs.py "$ACTUAL_LOG_DIR"

        # Analysis 2: run_attack_analysis.py
        echo "Running attack analysis..."
        python run_attack_analysis.py \
            "$ACTUAL_LOG_DIR/comprehensive_log_global.json" \
            --output "$ACTUAL_LOG_DIR/attack_analysis"

        echo "Analysis complete"
ENDSSH

    log "✓ Analysis complete"
}

# Download results
download_results() {
    log "Downloading results..."

    # Create local results directory with experiment ID (includes config name)
    RESULTS_DIR="$PROJECT_DIR/aws-results-${EXPERIMENT_ID}"
    mkdir -p "$RESULTS_DIR"

    # Download logs (with timeout options)
    # Use specific LOG_DIR instead of logs-* to avoid downloading all experiments
    log "Downloading experiment logs from: $LOG_DIR"
    rsync -avz --progress \
        -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=60" \
        ubuntu@"$PUBLIC_IP":"$REMOTE_DIR/$LOG_DIR/" "$RESULTS_DIR/" || warn "Failed to download some log files"

    # Download experiment log
    log "Downloading experiment log file..."
    scp -i "$KEY_PATH" -o ConnectTimeout=30 -o ServerAliveInterval=60 \
        ubuntu@"$PUBLIC_IP":"$REMOTE_DIR/experiment.log" \
        "$RESULTS_DIR/" || warn "Failed to download experiment.log"

    log "✓ Results downloaded to: $RESULTS_DIR"
}

# Terminate instance (spot instances can't be stopped, only terminated)
terminate_instance() {
    log "Terminating EC2 instance..."

    aws ec2 terminate-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'TerminatingInstances[0].CurrentState.Name' \
        --output text

    log "✓ Instance terminated: $INSTANCE_ID"
}

# Cleanup on error
cleanup_on_error() {
    error "Script failed. Cleaning up..."

    if [ -n "$INSTANCE_ID" ]; then
        warn "Terminating instance: $INSTANCE_ID"
        aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" || true
    fi
}

# ==================== MAIN EXECUTION ====================

# Run a single experiment with the given config file
run_single_experiment() {
    local config_file="$1"
    local skip_setup="${2:-false}"  # Skip instance setup if already done

    # Set config-specific variables
    CONFIG_FILE="$config_file"
    CONFIG_FILE_FULL_PATH="$PROJECT_DIR/$CONFIG_FILE"
    EXPERIMENT_NAME=$(basename -- "$CONFIG_FILE" .yml)

    log "=========================================="
    log "Running Experiment: $EXPERIMENT_NAME"
    log "=========================================="
    log "Config file: $CONFIG_FILE"
    log ""

    # Check if this specific config file exists
    check_config_file "$CONFIG_FILE_FULL_PATH"

    # Generate unique experiment ID
    EXPERIMENT_ID=$(generate_experiment_id "$EXPERIMENT_NAME")
    log "Experiment ID: $EXPERIMENT_ID"
    echo ""

    # Only do setup if this is the first experiment
    if [ "$skip_setup" = "false" ]; then
        # Trap errors
        trap cleanup_on_error ERR

        # Setup AWS resources
        setup_security_group
        get_ami_id

        # Launch instance
        launch_instance

        # Setup instance
        setup_instance

        # Upload code
        upload_code

        # Install dependencies
        install_dependencies
    fi

    # Save experiment metadata
    save_experiment_metadata "$EXPERIMENT_ID" "$INSTANCE_ID" "$CONFIG_FILE" "$EXPERIMENT_NAME" "$PUBLIC_IP" "running"

    # Run experiment
    run_experiment

    # Monitor experiment
    monitor_experiment

    # Run analysis
    run_analysis

    # Download results
    download_results

    # Update experiment status to completed
    update_experiment_status "$EXPERIMENT_ID" "completed"

    log ""
    log "=========================================="
    log "✓ EXPERIMENT COMPLETE!"
    log "=========================================="
    log "Experiment ID: $EXPERIMENT_ID"
    log "Results are in: $RESULTS_DIR"
    log ""

    # Store result for summary
    COMPLETED_EXPERIMENTS+=("$EXPERIMENT_ID|$RESULTS_DIR")
}

# Main function to run all experiments
main() {
    log "=========================================="
    log "AWS EC2 Multi-Experiment Runner"
    log "=========================================="
    log "Number of experiments to run: ${#CONFIG_FILES[@]}"
    if [ "$REUSE_INSTANCE" = "true" ]; then
        log "Mode: Single instance (reuse for all experiments)"
    else
        log "Mode: Multiple instances (one per experiment)"
    fi
    log ""

    # Check general requirements once
    check_requirements

    # Array to track completed experiments
    COMPLETED_EXPERIMENTS=()

    # Run each experiment
    for i in "${!CONFIG_FILES[@]}"; do
        local config_file="${CONFIG_FILES[$i]}"
        local experiment_num=$((i + 1))
        local skip_setup="false"

        log ""
        log "######################################"
        log "# EXPERIMENT $experiment_num of ${#CONFIG_FILES[@]}: $config_file"
        log "######################################"
        log ""

        # If reusing instance and this is not the first experiment, skip setup
        if [ "$REUSE_INSTANCE" = "true" ] && [ $i -gt 0 ]; then
            skip_setup="true"
            log "Reusing existing instance: $INSTANCE_ID ($PUBLIC_IP)"
            log ""
        fi

        # Run the experiment
        run_single_experiment "$config_file" "$skip_setup"

        log ""
        log "✓ Completed experiment $experiment_num of ${#CONFIG_FILES[@]}"
        log ""
    done

    # Terminate instance if we were reusing it
    if [ "$REUSE_INSTANCE" = "true" ] && [ -n "$INSTANCE_ID" ]; then
        log "All experiments complete. Terminating shared instance..."
        terminate_instance
    fi

    # Print final summary
    log ""
    log "=========================================="
    log "✓ ALL EXPERIMENTS COMPLETE!"
    log "=========================================="
    log "Total experiments run: ${#COMPLETED_EXPERIMENTS[@]}"
    log ""
    log "Results summary:"
    for result in "${COMPLETED_EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_id results_dir <<< "$result"
        log "  - $exp_id: $results_dir"
    done
    log ""
    log "Use: ./check_experiments.sh to manage all experiments"
}

# Run main function
main "$@"
