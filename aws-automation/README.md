# AWS Automation Scripts

This folder contains scripts for automating persona red-teaming experiments and deploying the user research application on AWS EC2.

## Scripts

### User Research Application Deployment

#### `deploy_user_research.sh`
Deploys the persona red-teaming web application (frontend + backend) on AWS EC2 for user research and data collection.

Features:
- Launches EC2 instance with Docker
- Deploys frontend (React) and backend (FastAPI) containers
- Sets up SQLite database for collecting user data
- Configures security groups for web access
- Returns public URLs for immediate use

**Usage:**
```bash
./aws-automation/deploy_user_research.sh
```

**What it does:**
1. Launches t3.small EC2 instance (Ubuntu 22.04)
2. Installs Docker and Docker Compose
3. Uploads application code
4. Builds and starts containers
5. Provides public URLs and SSH access

**Cost:** ~$0.02/hour (~$15/month for t3.small)

#### `manage_user_research.sh`
Management tool for the deployed user research application.

**Usage:**
```bash
./aws-automation/manage_user_research.sh [command]
```

**Commands:**
- `status` - Show deployment status and URLs
- `logs` - View application logs
- `restart` - Restart the application
- `stop` - Stop application containers
- `start` - Start application containers
- `ssh` - SSH into the instance
- `backup-db` - Backup database to local machine
- `update` - Update application code and restart
- `instance-stop` - Stop EC2 instance (saves costs)
- `instance-start` - Start EC2 instance
- `terminate` - Terminate instance (WARNING: deletes everything)

**Examples:**
```bash
# Check application status
./aws-automation/manage_user_research.sh status

# View live logs
./aws-automation/manage_user_research.sh logs

# Backup database
./aws-automation/manage_user_research.sh backup-db

# Update application code
./aws-automation/manage_user_research.sh update
```

---

### Experiment Automation

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

### User Research Deployment
- **`.aws-experiments.json`** - Deployment tracking database (created in project root)
- **`.last-deployment.txt`** - Quick reference for last deployment info
- **`database-backups/`** - Database backup files (when using `backup-db` command)

### Experiment Automation
- **`.aws-experiments.json`** - Experiment tracking database (created in project root)
- **`aws-results-{exp-id}/`** - Downloaded results (created in project root)

## See Also

For complete documentation, see `run_on_aws_guide.md` in the project root.
