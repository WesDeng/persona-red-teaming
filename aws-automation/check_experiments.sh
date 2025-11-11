#!/bin/bash

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

#
# Check status of running AWS experiments
# Usage: ./check_experiments.sh [experiment_id]
# Example: ./check_experiments.sh exp-qwen-*
#

# Load shared library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aws-lib.sh"

# Configuration
AUTO_UPDATE_SG=true  # Automatically update security group if IP changed

# Print header
print_header() {
    echo ""
    echo "=========================================="
    echo "AWS Experiments Status Checker"
    echo "=========================================="
    echo ""
}

# Print experiment status
print_experiment_status() {
    local exp_data="$1"

    local exp_id=$(echo "$exp_data" | jq -r '.id')
    local instance_id=$(echo "$exp_data" | jq -r '.instance_id')
    local exp_name=$(echo "$exp_data" | jq -r '.experiment_name')
    local config=$(echo "$exp_data" | jq -r '.config_file')
    local public_ip=$(echo "$exp_data" | jq -r '.public_ip')
    local status=$(echo "$exp_data" | jq -r '.status')
    local start_time=$(echo "$exp_data" | jq -r '.start_time')
    local region=$(echo "$exp_data" | jq -r '.region')

    echo ""
    echo "┌─────────────────────────────────────────────────────────────"
    echo "│ Experiment: $exp_name"
    echo "│ ID: $exp_id"
    echo "├─────────────────────────────────────────────────────────────"

    # Check instance status from AWS
    local aws_status=$(get_instance_status "$instance_id")
    local time_running=$(time_since "$start_time")

    echo "│ Instance ID: $instance_id"
    echo "│ AWS Status:  $aws_status"
    echo "│ Region:      $region"
    echo "│ Started:     $start_time ($time_running ago)"
    echo "│ Config:      $config"

    # If instance not found or terminated
    if [ "$aws_status" = "not-found" ] || [ "$aws_status" = "terminated" ]; then
        echo "│ Status:      ❌ TERMINATED (instance no longer exists)"
        echo "└─────────────────────────────────────────────────────────────"
        echo ""
        echo "  This instance has been terminated. Remove from tracking? [y/N]"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            remove_experiment "$exp_id"
            lib_log "Removed $exp_id from tracking"
        fi
        return
    fi

    # If instance is stopped
    if [ "$aws_status" = "stopped" ]; then
        echo "│ Status:      ⏸️  STOPPED"
        echo "└─────────────────────────────────────────────────────────────"
        echo ""
        echo "  Options: [1] Start instance  [2] Terminate  [3] Remove from tracking  [4] Skip"
        read -r option
        case $option in
            1)
                lib_log "Starting instance..."
                aws ec2 start-instances --region "$region" --instance-ids "$instance_id" >/dev/null
                lib_log "Instance starting..."
                ;;
            2)
                lib_log "Terminating instance..."
                aws ec2 terminate-instances --region "$region" --instance-ids "$instance_id" >/dev/null
                remove_experiment "$exp_id"
                lib_log "Instance terminated and removed from tracking"
                ;;
            3)
                remove_experiment "$exp_id"
                lib_log "Removed from tracking"
                ;;
        esac
        return
    fi

    # Instance is running - check experiment status
    if [ "$aws_status" = "running" ]; then
        # Update IP if changed
        local current_ip=$(get_instance_ip "$instance_id")
        if [ "$current_ip" != "$public_ip" ] && [ -n "$current_ip" ]; then
            update_experiment_ip "$exp_id" "$current_ip"
            public_ip="$current_ip"
            lib_debug "Updated IP to $current_ip"
        fi

        # Update security group if needed
        if [ "$AUTO_UPDATE_SG" = "true" ]; then
            local sg_id=$(aws ec2 describe-instances --region "$region" --instance-ids "$instance_id" --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text 2>/dev/null)
            if [ -n "$sg_id" ]; then
                update_security_group_ip "$sg_id"
            fi
        fi

        echo "│ Public IP:   $public_ip"

        # Check if tmux session is running
        local tmux_status=$(check_tmux_session "$public_ip")

        if [ "$tmux_status" = "running" ]; then
            echo "│ Status:      🔄 RUNNING"
            echo "├─────────────────────────────────────────────────────────────"
            echo "│ Recent logs:"
            echo "├─────────────────────────────────────────────────────────────"
            local logs=$(get_recent_logs "$public_ip" 5)
            echo "$logs" | while IFS= read -r line; do
                echo "│   $line"
            done
            echo "└─────────────────────────────────────────────────────────────"
            echo ""
            echo "  Options: [1] View full logs  [2] SSH to instance  [3] Skip"
            read -r option
            case $option in
                1)
                    ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@"$public_ip" \
                        "tail -n 50 /home/ubuntu/persona-red-teaming/experiment.log"
                    ;;
                2)
                    echo "Connecting to instance..."
                    echo "To attach to experiment: tmux attach -t redteam"
                    echo "To detach: Ctrl+B then D"
                    ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@"$public_ip"
                    ;;
            esac

        elif [ "$tmux_status" = "not-running" ]; then
            # Experiment has completed
            local exit_code=$(get_experiment_exit_code "$public_ip")

            if [ "$exit_code" = "0" ]; then
                echo "│ Status:      ✅ COMPLETED (exit code: 0)"
            elif [ "$exit_code" = "unknown" ]; then
                echo "│ Status:      ⚠️  COMPLETED (exit code unknown)"
            else
                echo "│ Status:      ❌ FAILED (exit code: $exit_code)"
            fi

            echo "├─────────────────────────────────────────────────────────────"
            echo "│ Last 10 lines of log:"
            echo "├─────────────────────────────────────────────────────────────"
            local logs=$(get_recent_logs "$public_ip" 10)
            echo "$logs" | while IFS= read -r line; do
                echo "│   $line"
            done
            echo "└─────────────────────────────────────────────────────────────"
            echo ""

            if [ "$status" != "completed" ]; then
                echo "  Options: [1] Run analysis & download  [2] Download only  [3] Terminate instance  [4] Skip"
                read -r option
                case $option in
                    1)
                        # Find log directory
                        local log_dir=$(ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@"$public_ip" \
                            "find /home/ubuntu/persona-red-teaming/logs-* -type d -name harmbench | head -1" 2>/dev/null)

                        if [ -n "$log_dir" ]; then
                            run_remote_analysis "$public_ip" "$log_dir"
                            local results_path=$(download_experiment_results "$public_ip" "$exp_id")
                            update_experiment_status "$exp_id" "completed"

                            echo ""
                            echo "  Terminate instance now? [Y/n]"
                            read -r terminate
                            if [[ ! "$terminate" =~ ^[Nn]$ ]]; then
                                aws ec2 terminate-instances --region "$region" --instance-ids "$instance_id" >/dev/null
                                lib_log "Instance terminated"
                                update_experiment_status "$exp_id" "terminated"
                            fi
                        else
                            lib_error "Could not find log directory"
                        fi
                        ;;
                    2)
                        local results_path=$(download_experiment_results "$public_ip" "$exp_id")
                        update_experiment_status "$exp_id" "completed"
                        ;;
                    3)
                        aws ec2 terminate-instances --region "$region" --instance-ids "$instance_id" >/dev/null
                        lib_log "Instance terminated"
                        update_experiment_status "$exp_id" "terminated"
                        ;;
                esac
            else
                echo "  Status already marked as completed"
                echo "  Terminate instance? [y/N]"
                read -r terminate
                if [[ "$terminate" =~ ^[Yy]$ ]]; then
                    aws ec2 terminate-instances --region "$region" --instance-ids "$instance_id" >/dev/null
                    lib_log "Instance terminated"
                    update_experiment_status "$exp_id" "terminated"
                fi
            fi
        else
            echo "│ Status:      ⚠️  UNKNOWN (cannot check tmux)"
            echo "└─────────────────────────────────────────────────────────────"
        fi
    fi
}

# Main function
main() {
    print_header

    # Check if experiments file exists
    if [ ! -f "$EXPERIMENTS_FILE" ]; then
        lib_warn "No experiments tracked yet"
        lib_log "Run ./deploy_and_run.sh to start an experiment"
        exit 0
    fi

    # Get filter pattern if provided
    local filter_pattern="${1:-*}"

    # Get all experiments
    local experiments=$(list_all_experiments)

    if [ -z "$experiments" ] || [ "$experiments" = "null" ]; then
        lib_warn "No experiments found"
        exit 0
    fi

    local count=0
    local matched=0

    # Process each experiment
    while IFS= read -r exp; do
        count=$((count + 1))
        local exp_id=$(echo "$exp" | jq -r '.id')

        # Apply filter if provided
        if [ "$filter_pattern" != "*" ]; then
            if [[ ! "$exp_id" == $filter_pattern ]]; then
                continue
            fi
        fi

        matched=$((matched + 1))
        print_experiment_status "$exp"
    done <<< "$experiments"

    echo ""
    if [ $matched -eq 0 ]; then
        lib_warn "No experiments matched pattern: $filter_pattern"
    else
        lib_log "Checked $matched experiment(s)"
    fi

    echo ""
}

# Run main function
main "$@"
