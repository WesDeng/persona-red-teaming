# AWS Deployment Guide

Fully automated deployment of persona red-teaming experiments on AWS EC2 with support for managing multiple parallel experiments.

## Modular Workflow

The AWS deployment system consists of three modular scripts (located in `aws-automation/`):

1. **`aws-automation/aws-setup.sh`** - One-time setup: Install prerequisites, configure AWS credentials, verify access
2. **`aws-automation/deploy_and_run.sh`** - Deploy and run: Launch instances, run experiments, auto-download results
3. **`aws-automation/check_experiments.sh`** - Manage experiments: Check status, run analysis, download results for multiple experiments

This allows:
- Run multiple experiments in parallel
- Check on experiments at any time (even after network disconnection)
- Resume management of experiments from any machine with AWS access

## Features

✅ Automatic EC2 instance provisioning
✅ Code upload and dependency installation
✅ Experiment execution with monitoring
✅ Automatic analysis (both scripts)
✅ Results download
✅ Instance shutdown when complete
✅ Support for Spot Instances (70% cost savings)
✅ **Multiple parallel experiment tracking**
✅ **Automatic IP change handling**
✅ **Experiment state persistence**

## Quick Start

### One-Time Setup

1. **Install prerequisites and configure AWS:**

```bash
chmod +x aws-automation/aws-setup.sh
./aws-automation/aws-setup.sh
```

This will:
- Install AWS CLI and jq
- Configure AWS credentials (you'll enter Access Key ID and Secret Access Key)
- **Verify credentials work and EC2 access is available**
- **Check for .env file with API keys**
- Create an EC2 key pair

The script will confirm:
```
✓ AWS credentials verified
  Account ID
  User/Role:
✓ EC2 access verified
✓ .env file found
✓ API keys found in .env
```

2. **If .env doesn't exist, create it:**

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...
EOF
```

### Run an Experiment

**Single command to run everything:**

```bash
chmod +x aws-automation/*.sh
./aws-automation/deploy_and_run.sh configs/qwen-target.yml
```

This will automatically:
1. Generate unique experiment ID (e.g., `exp-qwen-target-20251108-141706`)
2. Launch EC2 instance (Spot instance by default)
3. Tag instance with experiment name
4. **Save experiment metadata to `.aws-experiments.json`**
5. Upload your code
6. Install dependencies
7. Run the experiment in tmux (persists after disconnect)
8. Monitor progress (prints updates every 60 seconds)
9. Run both analysis scripts when complete
10. Download all results to `aws-results-exp-qwen-target-20251108-141706/` (includes config name)
11. Stop the instance (can be terminated manually later)

**Running multiple experiments in parallel:**

You can launch multiple experiments at once - each gets its own instance and unique ID:

```bash
# Launch all in separate terminals or background
./aws-automation/deploy_and_run.sh configs/qwen-target.yml &
./aws-automation/deploy_and_run.sh configs/rainbow-RT1.yml &
./aws-automation/deploy_and_run.sh configs/rainbow-User1.yml &
```

All experiments are tracked in `.aws-experiments.json` and can be managed with `aws-automation/check_experiments.sh`.

### What Happens During Execution

```
[2025-01-07 10:00:00] Checking requirements...
[2025-01-07 10:00:01] ✓ All requirements met
[2025-01-07 10:00:02] Setting up security group...
[2025-01-07 10:00:03] ✓ Security group created
[2025-01-07 10:00:04] Launching EC2 instance (t3.medium)...
[2025-01-07 10:00:15] ✓ Instance launched: i-1234567890abcdef0
[2025-01-07 10:00:45] ✓ Instance running at: 54.123.45.67
[2025-01-07 10:01:00] ✓ SSH is ready
[2025-01-07 10:01:05] Uploading code to instance...
[2025-01-07 10:01:30] ✓ Code uploaded
[2025-01-07 10:01:35] Installing Python dependencies...
[2025-01-07 10:05:00] ✓ Dependencies installed
[2025-01-07 10:05:05] Starting experiment: configs/qwen-target.yml
[2025-01-07 10:05:10] ✓ Experiment started
[2025-01-07 10:05:15] Monitoring experiment progress...
[2025-01-07 10:05:15] This may take several hours...

[Updates every 60 seconds with recent log output]

[Hours later...]
[2025-01-07 18:30:45] ✓ Experiment completed!
[2025-01-07 18:30:50] Running analysis scripts...
[2025-01-07 18:35:00] ✓ Analysis complete
[2025-01-07 18:35:05] Downloading results...
[2025-01-07 18:36:00] ✓ Results downloaded to: ./aws-results-20250107-183600
[2025-01-07 18:36:05] Stopping EC2 instance...
[2025-01-07 18:36:10] ✓ Instance stopped: i-1234567890abcdef0
[2025-01-07 18:36:15] ✓ ALL DONE!
```

### Managing Multiple Experiments

Use `aws-automation/check_experiments.sh` to view and manage all your experiments:

```bash
./aws-automation/check_experiments.sh
```

**Features:**
- Lists all tracked experiments with current status
- Automatically detects if your IP changed and updates security group
- Shows real-time status from AWS (running, stopped, terminated)
- Displays recent logs for running experiments
- Detects experiment completion by checking tmux session
- Reports exit codes (success/failure)
- Interactive options for completed experiments

**Example output:**

```
==========================================
AWS Experiments Status Checker
==========================================

┌─────────────────────────────────────────────────────────────
│ Experiment: qwen-target
│ ID: exp-qwen-target-20251108-141706
├─────────────────────────────────────────────────────────────
│ Instance ID: i-07c072ac6ca4b7afd
│ AWS Status:  running
│ Region:      us-west-2
│ Started:     2025-11-08T14:17:06Z (3h 25m ago)
│ Config:      configs/qwen-target.yml
│ Public IP:   34.212.174.184
│ Status:      🔄 RUNNING
├─────────────────────────────────────────────────────────────
│ Recent logs:
├─────────────────────────────────────────────────────────────
│   Processing behavior 125/150...
│   ASR so far: 28.5%
│   Generated 450 prompts
└─────────────────────────────────────────────────────────────

  Options: [1] View full logs  [2] SSH to instance  [3] Skip
```

**When experiment completes:**

The script detects completion and offers:
1. **Run analysis & download** - Executes both analysis scripts remotely, downloads results
2. **Download only** - Just downloads results without running analysis
3. **Terminate instance** - Immediately terminates to stop charges
4. **Skip** - Leave for later

**Filter specific experiments:**

```bash
# Check only experiments matching a pattern
./aws-automation/check_experiments.sh exp-qwen-*
./aws-automation/check_experiments.sh exp-*-20251108-*
```

**When to use check_experiments.sh:**

- ✅ After launching experiments in background
- ✅ When your network disconnected during monitoring
- ✅ To check on experiments from a different machine
- ✅ To manage completed experiments (run analysis, download, cleanup)
- ✅ When your IP changed and SSH stopped working (auto-fixes security group)

**State file (`.aws-experiments.json`):**

All experiment metadata is stored locally in `.aws-experiments.json`:

```json
{
  "experiments": [
    {
      "id": "exp-qwen-target-20251108-141706",
      "instance_id": "i-07c072ac6ca4b7afd",
      "config_file": "configs/qwen-target.yml",
      "experiment_name": "qwen-target",
      "start_time": "2025-11-08T14:17:06Z",
      "public_ip": "34.212.174.184",
      "status": "running",
      "region": "us-west-2"
    }
  ]
}
```

This file is automatically managed by the scripts. You can commit it to git to share experiment tracking with your team (just ensure IPs/credentials are not sensitive).

## Complete Workflow Example

Here's a typical workflow for running multiple experiments:

**1. One-time setup (first time only):**
```bash
./aws-automation/aws-setup.sh
# Enter AWS credentials when prompted
# Verify everything is configured correctly
```

**2. Launch multiple experiments:**
```bash
# Terminal 1
./aws-automation/deploy_and_run.sh configs/qwen-target.yml

# Terminal 2
./aws-automation/deploy_and_run.sh configs/rainbow-RT1.yml

# Or run in background
./aws-automation/deploy_and_run.sh configs/rainbow-User1.yml &
```

Each experiment gets a unique ID and runs independently.

**3. Close your laptop / disconnect (optional):**

Experiments continue running on AWS. The tmux session keeps them alive.

**4. Check status later (from anywhere):**
```bash
./aws-automation/check_experiments.sh

# Output shows all 3 experiments:
# - exp-qwen-target-20251108-141706 → 🔄 RUNNING (2h 15m ago)
# - exp-rainbow-RT1-20251108-143522 → 🔄 RUNNING (1h 58m ago)
# - exp-rainbow-User1-20251108-150103 → ✅ COMPLETED (exit code: 0)
```

**5. For completed experiments:**

The script prompts you:
```
Options: [1] Run analysis & download  [2] Download only  [3] Terminate  [4] Skip
```

Choose option 1 to run analysis and download results automatically.

**6. Clean up when done:**
```bash
# Terminate all instances
aws ec2 terminate-instances --instance-ids \
  i-07c072ac6ca4b7afd \
  i-08d183bc7db5c8bge \
  i-09e294cd8ec6d9chf
```

**Key benefits of this workflow:**
- ✅ No need to keep laptop on
- ✅ Experiments continue even if SSH disconnects
- ✅ Can manage from any machine with AWS credentials
- ✅ All experiments tracked in one place
- ✅ Automatic IP change handling
- ✅ One command to check all experiments

## Configuration

Edit `aws-automation/deploy_and_run.sh` to customize:

```bash
# AWS Configuration
AWS_REGION="us-east-2"              # Default region (change if needed)
INSTANCE_TYPE="t3.medium"           # Instance size (t3.medium is good for API-based)
SPOT_INSTANCE=true                  # Use spot instances (70% cheaper)

# Key Configuration
KEY_NAME="persona-redteaming-key"   # Must match your EC2 key pair name
```

## Cost Estimation

### Instance Costs (per hour)

| Instance Type | On-Demand | Spot (typical) | Recommended For |
|---------------|-----------|----------------|-----------------|
| t3.medium     | $0.0416   | ~$0.0125       | Most experiments |
| t3.large      | $0.0832   | ~$0.0250       | Larger workloads |
| t3.xlarge     | $0.1664   | ~$0.0500       | Heavy processing |

**For a 10-hour experiment on t3.medium Spot:** ~$0.13

### API Costs

API costs depend on your experiment configuration:
- **Together AI (Qwen):** ~$0.20 per 1M tokens
- **OpenAI (GPT-4o):** ~$2.50 per 1M input tokens, ~$10 per 1M output tokens

Estimate total cost based on your `max_iters` × `num_samples`.

## Advanced Usage

### Run Multiple Experiments in Sequence

```bash
# Create a batch script
cat > run_all_experiments.sh << 'EOF'
#!/bin/bash
./deploy_and_run.sh configs/qwen-target.yml
./deploy_and_run.sh configs/rainbow-RT1.yml
./deploy_and_run.sh configs/rainbow-User1.yml
EOF

chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

### Keep Instance Running (Don't Auto-Stop)

Comment out the stop_instance call in `deploy_and_run.sh`:

```bash
# stop_instance  # Comment this out
```

Then manually stop later:
```bash
aws ec2 stop-instances --instance-ids i-1234567890abcdef0
```

### SSH into Running Instance

```bash
# The script prints the instance IP
ssh -i ~/.ssh/persona-redteaming-key.pem ubuntu@<INSTANCE_IP>

# Attach to tmux session
tmux attach -t redteam
```

### Monitor Experiment Remotely

```bash
# Watch logs in real-time
ssh -i ~/.ssh/persona-redteaming-key.pem ubuntu@<INSTANCE_IP> \
    "tail -f /home/ubuntu/persona-red-teaming/experiment.log"
```

### Download Results While Running

```bash
# Sync logs periodically while experiment runs
rsync -avz -e "ssh -i ~/.ssh/persona-redteaming-key.pem" \
    ubuntu@<INSTANCE_IP>:/home/ubuntu/persona-red-teaming/logs-*/ \
    ./partial-results/
```

## Troubleshooting

### IP Address Changed / SSH Stopped Working

If your IP address changes (common on laptops, home networks):

**Automatic fix:**
```bash
./aws-automation/check_experiments.sh
# Script auto-detects IP change and updates security group
```

**Manual fix:**
```bash
# Get your current IP
MY_IP=$(curl -s https://checkip.amazonaws.com)

# Add to security group
aws ec2 authorize-security-group-ingress \
    --region us-west-2 \
    --group-id sg-XXXXXXXXX \
    --protocol tcp --port 22 --cidr ${MY_IP}/32
```

### Lost Track of Experiments

If you lost your `.aws-experiments.json` file:

```bash
# List all running instances
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=persona-redteaming-*" \
    --query 'Reservations[*].Instances[*].[InstanceId,PublicIpAddress,Tags[?Key==`Name`].Value|[0],State.Name]' \
    --output table
```

Then manually recreate `.aws-experiments.json` or SSH directly to instances using their IPs.

### Experiment Completed But Wasn't Detected

If `check_experiments.sh` shows experiment still running but it actually finished:

```bash
# SSH to instance and check
ssh -i ~/.ssh/persona-redteaming-key.pem ubuntu@<IP>

# Check tmux sessions
tmux list-sessions

# If no 'redteam' session, experiment is done
# Check exit code
cat /home/ubuntu/persona-red-teaming/experiment_exit_code.txt
```

### Multiple Experiments on Same Config

You can run the same config multiple times - each gets a unique ID with timestamp:
- `exp-qwen-target-20251108-141706`
- `exp-qwen-target-20251108-153022`
- `exp-qwen-target-20251108-181545`

They won't interfere with each other.

### "Key pair not found"

Create the key pair:
```bash
aws ec2 create-key-pair \
    --key-name persona-redteaming-key \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/persona-redteaming-key.pem
chmod 400 ~/.ssh/persona-redteaming-key.pem
```

### "AMI not found in region"

The script auto-detects the correct Ubuntu AMI. If it fails, manually find one:
```bash
aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId'
```

### "Spot instance request failed"

Switch to on-demand:
```bash
# In deploy_and_run.sh
SPOT_INSTANCE=false
```

### Experiment failed mid-run

The instance is left running (stopped). To clean up:
```bash
# List your instances
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=persona-redteaming-*" \
    --query 'Reservations[*].Instances[*].[InstanceId,State.Name]'

# Terminate specific instance
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

### Check script logs

All output is printed to terminal. To save:
```bash
./deploy_and_run.sh configs/qwen-target.yml 2>&1 | tee deployment.log
```

## Security Best Practices

1. **Never commit `.env` or `.pem` files** to git (already in `.gitignore`)
2. **Rotate API keys** regularly
3. **Delete stopped instances** when completely done:
   ```bash
   aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
   ```
4. **Review security group rules** periodically:
   ```bash
   aws ec2 describe-security-groups --group-names persona-redteaming-sg
   ```

## Manual Cleanup

When you're completely done with all experiments:

```bash
# Terminate all persona-redteaming instances
aws ec2 terminate-instances \
    --instance-ids $(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=persona-redteaming-*" \
        --query 'Reservations[*].Instances[*].InstanceId' \
        --output text)

# Delete security group
aws ec2 delete-security-group --group-name persona-redteaming-sg

# Delete key pair (optional)
aws ec2 delete-key-pair --key-name persona-redteaming-key
rm ~/.ssh/persona-redteaming-key.pem
```

## Support

For issues with:
- **AWS setup**: Check AWS documentation or run `aws configure`
- **Script errors**: Review the error message and check prerequisites
- **Experiment issues**: Same debugging as local runs

## What Gets Downloaded

After completion, results are in `aws-results-exp-{config-name}-{timestamp}/`:

**Example for `qwen-target.yml`:**
```
aws-results-exp-qwen-target-20251108-141706/
├── Qwen/
│   └── Qwen2.5-7B-Instruct-Turbo/
│       └── harmbench/
│           ├── comprehensive_log_global.json
│           ├── rainbowplus_log_global.json
│           ├── analysis_results.json
│           └── attack_analysis/
│               ├── summary.json
│               └── [other analysis files]
└── experiment.log
```

**Example for `rainbow-RT1.yml`:**
```
aws-results-exp-rainbow-RT1-20251108-143522/
├── [model folders...]
└── experiment.log
```

The folder name includes the config name, making it easy to identify which experiment the results belong to!
