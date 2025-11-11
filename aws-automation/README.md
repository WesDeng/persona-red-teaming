# AWS Automation Scripts

This folder contains scripts for automating persona red-teaming experiments on AWS EC2.

## Scripts

### `aws-setup.sh`
One-time setup script that:
- Installs AWS CLI and jq
- Configures AWS credentials
- Verifies EC2 access
- Checks for .env file with API keys
- Creates EC2 key pair

**Usage:**
```bash
./aws-automation/aws-setup.sh
```

### `deploy_and_run.sh`
Main deployment script that:
- Launches EC2 instance (spot instance by default)
- Uploads code and dependencies
- Runs experiment in tmux session
- Monitors progress
- Downloads results when complete
- Saves experiment metadata to `.aws-experiments.json`

**Usage:**
```bash
./aws-automation/deploy_and_run.sh <config-file>
```

**Example:**
```bash
./aws-automation/deploy_and_run.sh configs/qwen-target.yml
```

### `check_experiments.sh`
Interactive experiment manager that:
- Lists all running experiments
- Shows real-time AWS status
- Displays recent logs
- Auto-fixes IP changes
- Offers options for completed experiments (run analysis, download, terminate)

**Usage:**
```bash
# Check all experiments
./aws-automation/check_experiments.sh

# Check specific experiments
./aws-automation/check_experiments.sh exp-qwen-*
```

### `aws-lib.sh`
Shared library with common functions. Do not run directly - it's sourced by other scripts.

Contains functions for:
- Experiment metadata management
- AWS instance operations
- Remote SSH operations
- Analysis and result downloads

## Configuration

Default settings in `deploy_and_run.sh`:
- **Region:** us-east-2
- **Instance Type:** t3.medium
- **Spot Instances:** Enabled (70% cost savings)

Edit `deploy_and_run.sh` to customize these settings.

## Files Created

- **`.aws-experiments.json`** - Experiment tracking database (created in project root)
- **`aws-results-{exp-id}/`** - Downloaded results (created in project root)

## See Also

For complete documentation, see `run_on_aws_guide.md` in the project root.
