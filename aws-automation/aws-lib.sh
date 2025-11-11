#!/bin/bash

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

#
# Shared library functions for AWS EC2 experiment management
# Source this file in other scripts: source aws-lib.sh
#

# Configuration
AWS_REGION="${AWS_REGION:-us-east-2}"
KEY_NAME="${KEY_NAME:-persona-redteaming-key}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/${KEY_NAME}.pem}"

# Get the project root directory (parent of aws-automation folder)
LIB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$LIB_SCRIPT_DIR/.." && pwd)"
EXPERIMENTS_FILE="$PROJECT_ROOT/.aws-experiments.json"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
lib_log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

lib_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

lib_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

lib_debug() {
    if [ "${DEBUG:-false}" = "true" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Initialize experiments file if it doesn't exist
init_experiments_file() {
    if [ ! -f "$EXPERIMENTS_FILE" ]; then
        echo '{"experiments":[]}' > "$EXPERIMENTS_FILE"
        lib_log "Created $EXPERIMENTS_FILE"
    fi
}

# Save experiment metadata
# Usage: save_experiment_metadata <exp_id> <instance_id> <config> <name> <ip> <status>
save_experiment_metadata() {
    local exp_id="$1"
    local instance_id="$2"
    local config_file="$3"
    local exp_name="$4"
    local public_ip="$5"
    local status="${6:-running}"

    init_experiments_file

    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Create new experiment entry
    local new_exp=$(cat <<EOF
{
  "id": "$exp_id",
  "instance_id": "$instance_id",
  "config_file": "$config_file",
  "experiment_name": "$exp_name",
  "start_time": "$timestamp",
  "public_ip": "$public_ip",
  "status": "$status",
  "region": "$AWS_REGION"
}
EOF
)

    # Add to experiments array using jq
    jq --argjson exp "$new_exp" '.experiments += [$exp]' "$EXPERIMENTS_FILE" > "$EXPERIMENTS_FILE.tmp"
    mv "$EXPERIMENTS_FILE.tmp" "$EXPERIMENTS_FILE"

    lib_log "Saved metadata for experiment: $exp_id"
}

# Update experiment status
# Usage: update_experiment_status <exp_id> <new_status>
update_experiment_status() {
    local exp_id="$1"
    local new_status="$2"

    if [ ! -f "$EXPERIMENTS_FILE" ]; then
        lib_error "Experiments file not found"
        return 1
    fi

    jq --arg id "$exp_id" --arg status "$new_status" \
        '(.experiments[] | select(.id == $id) | .status) = $status' \
        "$EXPERIMENTS_FILE" > "$EXPERIMENTS_FILE.tmp"
    mv "$EXPERIMENTS_FILE.tmp" "$EXPERIMENTS_FILE"

    lib_debug "Updated status for $exp_id to $new_status"
}

# Update experiment public IP
# Usage: update_experiment_ip <exp_id> <new_ip>
update_experiment_ip() {
    local exp_id="$1"
    local new_ip="$2"

    if [ ! -f "$EXPERIMENTS_FILE" ]; then
        lib_error "Experiments file not found"
        return 1
    fi

    jq --arg id "$exp_id" --arg ip "$new_ip" \
        '(.experiments[] | select(.id == $id) | .public_ip) = $ip' \
        "$EXPERIMENTS_FILE" > "$EXPERIMENTS_FILE.tmp"
    mv "$EXPERIMENTS_FILE.tmp" "$EXPERIMENTS_FILE"

    lib_debug "Updated IP for $exp_id to $new_ip"
}

# Load experiment by ID
# Usage: load_experiment <exp_id>
load_experiment() {
    local exp_id="$1"

    if [ ! -f "$EXPERIMENTS_FILE" ]; then
        lib_error "Experiments file not found"
        return 1
    fi

    jq --arg id "$exp_id" '.experiments[] | select(.id == $id)' "$EXPERIMENTS_FILE"
}

# List all experiments
# Usage: list_all_experiments
list_all_experiments() {
    if [ ! -f "$EXPERIMENTS_FILE" ]; then
        echo "[]"
        return 0
    fi

    jq '.experiments[]' "$EXPERIMENTS_FILE"
}

# Remove experiment from tracking
# Usage: remove_experiment <exp_id>
remove_experiment() {
    local exp_id="$1"

    if [ ! -f "$EXPERIMENTS_FILE" ]; then
        return 0
    fi

    jq --arg id "$exp_id" '.experiments |= map(select(.id != $id))' \
        "$EXPERIMENTS_FILE" > "$EXPERIMENTS_FILE.tmp"
    mv "$EXPERIMENTS_FILE.tmp" "$EXPERIMENTS_FILE"

    lib_log "Removed experiment $exp_id from tracking"
}

# Get instance status from AWS
# Usage: get_instance_status <instance_id>
get_instance_status() {
    local instance_id="$1"

    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null || echo "not-found"
}

# Get instance public IP from AWS
# Usage: get_instance_ip <instance_id>
get_instance_ip() {
    local instance_id="$1"

    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null || echo ""
}

# Check if tmux session is running on remote instance
# Usage: check_tmux_session <public_ip> <session_name>
check_tmux_session() {
    local public_ip="$1"
    local session_name="${2:-redteam}"

    local result=$(ssh -i "$KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        ubuntu@"$public_ip" \
        "tmux list-sessions 2>/dev/null | grep $session_name || echo 'none'" 2>/dev/null)

    if [ "$result" = "none" ] || [ -z "$result" ]; then
        echo "not-running"
    else
        echo "running"
    fi
}

# Get recent logs from remote instance
# Usage: get_recent_logs <public_ip> <num_lines>
get_recent_logs() {
    local public_ip="$1"
    local num_lines="${2:-10}"

    ssh -i "$KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        ubuntu@"$public_ip" \
        "tail -n $num_lines /home/ubuntu/persona-red-teaming/experiment.log 2>/dev/null" 2>/dev/null || echo "Logs not available"
}

# Get experiment exit code from remote instance
# Usage: get_experiment_exit_code <public_ip>
get_experiment_exit_code() {
    local public_ip="$1"

    ssh -i "$KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        ubuntu@"$public_ip" \
        "cat /home/ubuntu/persona-red-teaming/experiment_exit_code.txt 2>/dev/null" 2>/dev/null || echo "unknown"
}

# Update security group to allow current IP
# Usage: update_security_group_ip <security_group_id>
update_security_group_ip() {
    local sg_id="$1"
    local current_ip=$(curl -s https://checkip.amazonaws.com)

    # Check if rule already exists
    local existing=$(aws ec2 describe-security-groups \
        --region "$AWS_REGION" \
        --group-ids "$sg_id" \
        --query "SecurityGroups[0].IpPermissions[?FromPort==\`22\`].IpRanges[?CidrIp==\`${current_ip}/32\`]" \
        --output text 2>/dev/null)

    if [ -z "$existing" ]; then
        lib_log "Adding current IP ($current_ip) to security group..."
        aws ec2 authorize-security-group-ingress \
            --region "$AWS_REGION" \
            --group-id "$sg_id" \
            --protocol tcp \
            --port 22 \
            --cidr "${current_ip}/32" &>/dev/null || true
        lib_log "✓ IP added to security group"
    else
        lib_debug "Current IP already in security group"
    fi
}

# Run analysis scripts on remote instance
# Usage: run_remote_analysis <public_ip> <log_directory>
run_remote_analysis() {
    local public_ip="$1"
    local log_directory="$2"

    lib_log "Running analysis scripts on remote instance..."

    # Run comprehensive log analysis
    lib_log "Running comprehensive log analysis..."
    ssh -i "$KEY_PATH" \
        -o StrictHostKeyChecking=no \
        ubuntu@"$public_ip" \
        "cd /home/ubuntu/persona-red-teaming && source venv/bin/activate && python analyze_comprehensive_logs.py $log_directory" 2>&1 | grep -E "^(DIVERSITY|COMPREHENSIVE|Success rate|ASR|filter_pass_rate|iteration_ASR)" || true

    # Run attack analysis
    lib_log "Running attack analysis..."
    ssh -i "$KEY_PATH" \
        -o StrictHostKeyChecking=no \
        ubuntu@"$public_ip" \
        "cd /home/ubuntu/persona-red-teaming && source venv/bin/activate && python run_attack_analysis.py ${log_directory}/comprehensive_log_global.json --output ${log_directory}/attack_analysis" 2>&1 | tail -20 || true

    lib_log "✓ Analysis complete"
}

# Download experiment results
# Usage: download_experiment_results <public_ip> <exp_id>
download_experiment_results() {
    local public_ip="$1"
    local exp_id="$2"

    local results_dir="$PROJECT_ROOT/aws-results-${exp_id}"

    lib_log "Downloading results to $results_dir/..."

    mkdir -p "$results_dir"

    # Download logs
    rsync -avz --progress \
        -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no" \
        ubuntu@"$public_ip":/home/ubuntu/persona-red-teaming/logs-*/ "$results_dir/" 2>&1 | grep -E "receiving|sent|total size" || true

    # Download experiment log
    scp -i "$KEY_PATH" \
        -o StrictHostKeyChecking=no \
        ubuntu@"$public_ip":/home/ubuntu/persona-red-teaming/experiment.log \
        "$results_dir/" &>/dev/null || true

    lib_log "✓ Results downloaded to: $results_dir/"
    echo "$results_dir"
}

# Generate experiment ID
# Usage: generate_experiment_id <experiment_name>
generate_experiment_id() {
    local exp_name="$1"
    echo "exp-${exp_name}-$(date +%Y%m%d-%H%M%S)"
}

# Format time duration
# Usage: format_duration <seconds>
format_duration() {
    local seconds="$1"
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))

    if [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m"
    else
        echo "${minutes}m"
    fi
}

# Calculate time since
# Usage: time_since <iso_timestamp>
time_since() {
    local timestamp="$1"
    local start_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$timestamp" "+%s" 2>/dev/null || date "+%s")
    local now_epoch=$(date "+%s")
    local diff=$((now_epoch - start_epoch))
    format_duration $diff
}

lib_debug "AWS library loaded (region: $AWS_REGION)"
