# Quick Start: Deploy User Research App on AWS

This guide gets your persona red-teaming web application deployed on AWS in ~15 minutes.

## Prerequisites

1. AWS account
2. AWS CLI configured with credentials
3. Your OpenAI API key (or other LLM API keys)

## Step 1: First-Time Setup (One Time Only)

If this is your first time using the AWS automation scripts:

```bash
./aws-automation/aws-setup.sh
```

This will:
- Verify AWS CLI is installed
- Create an EC2 key pair
- Save the key to `~/.ssh/persona-redteaming-key.pem`

**IMPORTANT:** Make sure you have a `.env` file with your API keys:

```bash
# Create .env file if you don't have one
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-ai-key-here
EOF
```

## Step 2: Deploy Application

Run the deployment script:

```bash
./aws-automation/deploy_user_research.sh
```

**What happens:**
1. Creates security group with proper ports (22, 80, 8000)
2. Launches t3.small EC2 instance (~$15/month)
3. Installs Docker and dependencies
4. Uploads your code
5. Builds and starts containers
6. Returns public URLs

**Time:** ~10-15 minutes for first deployment

## Step 3: Access Your Application

After deployment completes, you'll see:

```
========================================
DEPLOYMENT COMPLETE!
========================================

Application URLs:
  Frontend: http://YOUR_PUBLIC_IP
  Backend API: http://YOUR_PUBLIC_IP:8000
  Health Check: http://YOUR_PUBLIC_IP:8000/health

SSH Access:
  ssh -i ~/.ssh/persona-redteaming-key.pem ubuntu@YOUR_PUBLIC_IP
```

**Open the frontend URL in your browser** - your application is live!

## Step 4: Manage Your Deployment

Use the management script for common tasks:

### Check Status
```bash
./aws-automation/manage_user_research.sh status
```

### View Logs
```bash
./aws-automation/manage_user_research.sh logs
```

### Backup Database
```bash
./aws-automation/manage_user_research.sh backup-db
```

This downloads the SQLite database to `database-backups/history_TIMESTAMP.db`

### Update Application Code
Made changes to your code? Deploy them:

```bash
./aws-automation/manage_user_research.sh update
```

### Stop Instance to Save Money
Not using it? Stop the instance:

```bash
./aws-automation/manage_user_research.sh instance-stop
```

This stops compute charges (~$0.02/hour). You only pay for storage (~$3/month).

### Start Instance Again
```bash
./aws-automation/manage_user_research.sh instance-start
```

**Note:** Public IP will change! You'll need to rebuild the frontend with the new IP.

### SSH into Server
```bash
./aws-automation/manage_user_research.sh ssh
```

### Terminate Everything
When you're completely done:

```bash
./aws-automation/manage_user_research.sh terminate
```

This creates a final database backup, then deletes the instance.

## Common Tasks

### Configure API Keys After Deployment

If you didn't have a `.env` file during deployment:

```bash
# SSH into server
./aws-automation/manage_user_research.sh ssh

# Edit .env file
nano ~/persona-red-teaming/.env

# Add your keys:
OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-key-here

# Save (Ctrl+X, Y, Enter)

# Restart application
cd ~/persona-red-teaming
docker-compose restart
```

### Query the Database

```bash
# SSH into server
./aws-automation/manage_user_research.sh ssh

# Open database
sqlite3 ~/persona-red-teaming/data/history.db

# Example queries:
sqlite> .tables
sqlite> SELECT COUNT(*) FROM sessions;
sqlite> SELECT COUNT(*) FROM prompts;
sqlite> .quit
```

### Download Database for Local Analysis

```bash
# Backup to local machine
./aws-automation/manage_user_research.sh backup-db

# Analyze locally
sqlite3 database-backups/history_TIMESTAMP.db
```

### View Application Logs

```bash
# Live logs (Ctrl+C to exit)
./aws-automation/manage_user_research.sh logs

# Or via SSH
./aws-automation/manage_user_research.sh ssh
cd ~/persona-red-teaming
docker-compose logs -f backend  # Backend logs
docker-compose logs -f frontend # Frontend logs
```

### Restart After Configuration Changes

```bash
./aws-automation/manage_user_research.sh restart
```

## Cost Management

### Current Costs
- **Running:** ~$0.02/hour (~$15/month for t3.small)
- **Stopped:** ~$3/month (30GB EBS storage only)
- **Terminated:** $0

### Free Tier Eligible

If you have AWS free tier (first 12 months):
- Use t2.micro instead of t3.small (edit `deploy_user_research.sh` line 19)
- 750 hours/month free
- 30GB EBS free

Edit the deployment script:
```bash
nano aws-automation/deploy_user_research.sh
# Change line 19:
INSTANCE_TYPE="t2.micro"  # Free tier eligible
```

**Warning:** t2.micro has only 1GB RAM, which may be tight for Docker. t3.small (2GB RAM) is recommended.

### Stop When Not Using

Save money by stopping the instance when not in use:

```bash
# Stop at night
./aws-automation/manage_user_research.sh instance-stop

# Start in morning
./aws-automation/manage_user_research.sh instance-start
```

## Troubleshooting

### Application Not Responding

```bash
# Check status
./aws-automation/manage_user_research.sh status

# View logs for errors
./aws-automation/manage_user_research.sh logs

# Restart application
./aws-automation/manage_user_research.sh restart
```

### Can't SSH

```bash
# Check instance is running
./aws-automation/manage_user_research.sh status

# Ensure key permissions
chmod 400 ~/.ssh/persona-redteaming-key.pem

# Check your IP is allowed
# The script auto-adds your IP, but if it changed, re-run:
./aws-automation/deploy_user_research.sh
```

### Frontend Shows API Errors

The frontend needs to know the backend IP. If you stopped/started the instance:

```bash
# Get new public IP
./aws-automation/manage_user_research.sh status

# SSH and update docker-compose.yml
./aws-automation/manage_user_research.sh ssh
cd ~/persona-red-teaming
nano docker-compose.yml
# Update VITE_API_BASE_URL to new IP
# Save and rebuild:
docker-compose down
docker-compose up -d --build
```

### Out of Disk Space

```bash
# SSH into server
./aws-automation/manage_user_research.sh ssh

# Check disk usage
df -h

# Clean Docker
docker system prune -a

# Or expand EBS volume in AWS console
```

## Next Steps

1. **Test the application:** Visit the frontend URL and create some prompts
2. **Monitor usage:** Use `manage_user_research.sh status` regularly
3. **Backup data:** Run `manage_user_research.sh backup-db` periodically
4. **Set billing alerts:** In AWS console, set up billing alerts for your budget

## Files Created

After deployment, these files are created locally:

- `.aws-experiments.json` - Tracks deployment metadata
- `.last-deployment.txt` - Quick reference for latest deployment
- `database-backups/` - Database backups when you run `backup-db`

These are git-ignored and won't be committed.

## Summary of Commands

```bash
# Deploy
./aws-automation/deploy_user_research.sh

# Check status
./aws-automation/manage_user_research.sh status

# View logs
./aws-automation/manage_user_research.sh logs

# Backup database
./aws-automation/manage_user_research.sh backup-db

# Update code
./aws-automation/manage_user_research.sh update

# SSH access
./aws-automation/manage_user_research.sh ssh

# Stop instance (save money)
./aws-automation/manage_user_research.sh instance-stop

# Start instance
./aws-automation/manage_user_research.sh instance-start

# Terminate (delete everything)
./aws-automation/manage_user_research.sh terminate
```

## Support

- AWS deployment scripts: See `aws-automation/README.md`
- Manual deployment: See `AWS_DEPLOYMENT.md`
- Application issues: Check logs with `manage_user_research.sh logs`
