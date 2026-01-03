# AWS EC2 Deployment Guide

This guide walks you through deploying the Persona Red Teaming application on AWS EC2 with full database persistence and control.

## Overview

- **Cost**: Free tier eligible (t2.micro/t3.micro for 12 months, 750 hours/month)
- **Database**: SQLite on persistent EBS volume
- **Access**: Direct SSH access to server and database
- **Architecture**: Docker Compose with backend + frontend containers

---

## Prerequisites

1. AWS Account (free tier eligible)
2. Your OpenAI API key (or other LLM API keys)
3. SSH client installed on your local machine
4. Basic terminal/command line knowledge

---

## Step 1: Launch EC2 Instance

### 1.1 Sign into AWS Console

1. Go to https://console.aws.amazon.com/
2. Navigate to **EC2** service
3. Click **Launch Instance**

### 1.2 Configure Instance

**Name and Tags:**
- Name: `persona-red-teaming`

**Application and OS Images:**
- **AMI**: Ubuntu Server 22.04 LTS (free tier eligible)
- Architecture: 64-bit (x86)

**Instance Type:**
- **Type**: `t2.micro` (1 vCPU, 1 GiB RAM) - Free tier eligible
- Alternative: `t3.micro` (better performance, also free tier)

**Key Pair (login):**
- Click **Create new key pair**
- Key pair name: `persona-rt-key`
- Key pair type: RSA
- Private key file format: `.pem` (for Mac/Linux) or `.ppk` (for Windows PuTTY)
- **IMPORTANT**: Download and save this file securely - you'll need it to SSH

**Network Settings:**
Click **Edit** and configure:

1. **Auto-assign public IP**: Enable
2. **Firewall (Security Groups)**: Create security group
   - Security group name: `persona-rt-sg`
   - Description: `Security group for Persona Red Teaming app`

3. **Inbound Security Rules** - Add these rules:

   | Type        | Protocol | Port Range | Source Type | Source      | Description           |
   |-------------|----------|------------|-------------|-------------|-----------------------|
   | SSH         | TCP      | 22         | My IP       | 0.0.0.0/0   | SSH access            |
   | HTTP        | TCP      | 80         | Anywhere    | 0.0.0.0/0   | Frontend access       |
   | Custom TCP  | TCP      | 8000       | Anywhere    | 0.0.0.0/0   | Backend API access    |

   **Security Note**: For production, restrict SSH (port 22) to your IP only. For the "My IP" source, you can select "My IP" from dropdown to auto-fill your current IP address.

**Configure Storage:**
- Size: 20 GiB (free tier includes up to 30 GiB)
- Volume Type: General Purpose SSD (gp3)
- Delete on Termination: Uncheck if you want to preserve data if instance is terminated

**Advanced Details:**
- Leave defaults (not needed for basic deployment)

### 1.3 Launch Instance

1. Review your configuration in the summary panel
2. Click **Launch instance**
3. Wait for instance state to show "Running" (takes 1-2 minutes)
4. Note down your **Public IPv4 address** - you'll need this

---

## Step 2: Connect to Your Instance

### 2.1 Set Permissions on Key File (Mac/Linux)

```bash
chmod 400 ~/Downloads/persona-rt-key.pem
```

### 2.2 SSH into Instance

```bash
ssh -i ~/Downloads/persona-rt-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>
```

Replace `<YOUR_EC2_PUBLIC_IP>` with your instance's public IP address.

**First time connecting**: You'll see a warning about host authenticity. Type `yes` to continue.

You should see a welcome message and be logged into your Ubuntu server.

---

## Step 3: Install Docker and Dependencies

Run these commands on your EC2 instance:

### 3.1 Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### 3.2 Install Docker

```bash
# Install Docker
sudo apt install -y docker.io

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group (avoid needing sudo)
sudo usermod -aG docker ubuntu

# Log out and back in for group changes to take effect
exit
```

SSH back in:
```bash
ssh -i ~/Downloads/persona-rt-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>
```

### 3.3 Install Docker Compose

```bash
# Install Docker Compose
sudo apt install -y docker-compose

# Verify installation
docker --version
docker-compose --version
```

### 3.4 Install Git

```bash
sudo apt install -y git
```

---

## Step 4: Deploy the Application

### 4.1 Clone Repository

```bash
cd ~
git clone https://github.com/WesDeng/persona-red-teaming.git
cd persona-red-teaming
```

### 4.2 Configure Environment Variables

Create a `.env` file with your API keys:

```bash
nano .env
```

Add your API keys (press `i` to enter insert mode):

```bash
# OpenAI API Key (required)
OPENAI_API_KEY=sk-your-openai-key-here

# Google AI API Key (optional, for Gemini models)
GOOGLE_API_KEY=your-google-ai-key-here

# Other configurations
DATABASE_PATH=/app/data/history.db
```

**Save and exit**: Press `Ctrl+X`, then `Y`, then `Enter`

### 4.3 Update Frontend API URL

Edit the docker-compose.yml to use your EC2 public IP:

```bash
nano docker-compose.yml
```

Find the frontend service and update the `VITE_API_BASE_URL`:

```yaml
frontend:
  ...
  environment:
    - VITE_API_BASE_URL=http://<YOUR_EC2_PUBLIC_IP>:8000
```

Replace `<YOUR_EC2_PUBLIC_IP>` with your actual EC2 public IP address.

**Save and exit**: Press `Ctrl+X`, then `Y`, then `Enter`

### 4.4 Create Data Directory

```bash
mkdir -p data
```

### 4.5 Start the Application

```bash
docker-compose up -d
```

This will:
- Build the Docker images (takes 5-10 minutes first time)
- Start backend on port 8000
- Start frontend on port 80
- Create SQLite database at `./data/history.db`

### 4.6 Check Status

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f

# View backend logs only
docker-compose logs -f backend

# View frontend logs only
docker-compose logs -f frontend
```

Press `Ctrl+C` to stop viewing logs.

---

## Step 5: Access Your Application

### 5.1 Open in Browser

- **Frontend**: `http://<YOUR_EC2_PUBLIC_IP>`
- **Backend API**: `http://<YOUR_EC2_PUBLIC_IP>:8000`
- **Health Check**: `http://<YOUR_EC2_PUBLIC_IP>:8000/health`

### 5.2 Test the Application

1. Open the frontend URL in your browser
2. Create a new persona
3. Generate prompts
4. Verify everything works

---

## Step 6: Database Management

### 6.1 Access Database

Your SQLite database is stored at `~/persona-red-teaming/data/history.db`

**Install SQLite CLI:**
```bash
sudo apt install -y sqlite3
```

**Query database:**
```bash
sqlite3 ~/persona-red-teaming/data/history.db
```

**Example queries:**
```sql
-- List all tables
.tables

-- View all sessions
SELECT * FROM sessions ORDER BY created_at DESC LIMIT 10;

-- Count prompts by verdict
SELECT verdict, COUNT(*)
FROM guard_results
GROUP BY verdict;

-- View recent prompts with verdicts
SELECT p.prompt_text, g.verdict, g.score
FROM prompts p
JOIN guard_results g ON p.id = g.prompt_id
ORDER BY p.created_at DESC
LIMIT 10;

-- Exit SQLite
.quit
```

### 6.2 Export Database

```bash
# Copy database to your local machine
scp -i ~/Downloads/persona-rt-key.pem \
  ubuntu@<YOUR_EC2_PUBLIC_IP>:~/persona-red-teaming/data/history.db \
  ~/Desktop/history.db
```

### 6.3 Backup Database

**Create automated backup script:**

```bash
nano ~/backup-db.sh
```

Add this content:

```bash
#!/bin/bash
BACKUP_DIR=~/backups
mkdir -p $BACKUP_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp ~/persona-red-teaming/data/history.db $BACKUP_DIR/history_$TIMESTAMP.db
echo "Database backed up to $BACKUP_DIR/history_$TIMESTAMP.db"

# Keep only last 7 backups
cd $BACKUP_DIR
ls -t history_*.db | tail -n +8 | xargs rm -f
```

Make it executable:
```bash
chmod +x ~/backup-db.sh
```

**Run backup manually:**
```bash
~/backup-db.sh
```

**Automate with cron (daily at 2 AM):**
```bash
crontab -e
```

Add this line:
```
0 2 * * * ~/backup-db.sh
```

---

## Step 7: Application Management

### 7.1 Stop Application

```bash
cd ~/persona-red-teaming
docker-compose down
```

### 7.2 Restart Application

```bash
cd ~/persona-red-teaming
docker-compose restart
```

### 7.3 Update Application

```bash
cd ~/persona-red-teaming

# Stop containers
docker-compose down

# Pull latest changes
git pull

# Rebuild and restart
docker-compose up -d --build
```

### 7.4 View Logs

```bash
# All logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend
```

### 7.5 Check Container Health

```bash
docker-compose ps
docker stats
```

---

## Step 8: Monitoring and Maintenance

### 8.1 Monitor Disk Usage

```bash
# Check overall disk usage
df -h

# Check database size
du -h ~/persona-red-teaming/data/history.db

# Check Docker disk usage
docker system df
```

### 8.2 Clean Up Docker Resources

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Clean everything (careful!)
docker system prune -a
```

### 8.3 Monitor Memory and CPU

```bash
# Real-time monitoring
htop

# Or use top
top

# Docker container stats
docker stats
```

### 8.4 Set Up CloudWatch (Optional)

AWS CloudWatch can monitor your instance automatically:
1. Go to EC2 Console
2. Select your instance
3. Click "Monitoring" tab
4. Enable detailed monitoring

---

## Step 9: Security Best Practices

### 9.1 Update Security Group

After testing, restrict SSH access to your IP only:
1. Go to EC2 Console
2. Click "Security Groups"
3. Select your security group
4. Edit inbound rules for SSH (port 22)
5. Change source from `0.0.0.0/0` to "My IP"

### 9.2 Enable Automatic Security Updates

```bash
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

Select "Yes" when prompted.

### 9.3 Set Up Firewall (UFW)

```bash
# Enable firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 8000/tcp
sudo ufw enable
sudo ufw status
```

### 9.4 Protect API Keys

Never commit `.env` file to git. The `.gitignore` already excludes it.

---

## Step 10: Domain Name Setup (Optional)

### 10.1 Register Domain

Register a domain with Route 53, Namecheap, or any domain registrar.

### 10.2 Point Domain to EC2

1. Go to AWS Route 53 (or your DNS provider)
2. Create an A record pointing to your EC2 public IP:
   - Name: `persona.yourdomain.com`
   - Type: A
   - Value: `<YOUR_EC2_PUBLIC_IP>`

### 10.3 Update Frontend Configuration

Edit `docker-compose.yml`:
```yaml
frontend:
  environment:
    - VITE_API_BASE_URL=http://persona.yourdomain.com:8000
```

Rebuild and restart:
```bash
docker-compose down
docker-compose up -d --build
```

### 10.4 Set Up SSL/HTTPS (Recommended)

Use Let's Encrypt for free SSL certificates:

```bash
# Install Certbot
sudo apt install -y certbot

# Get certificate (replace with your domain)
sudo certbot certonly --standalone -d persona.yourdomain.com
```

Update nginx configuration to use SSL (requires modifying frontend nginx.conf to add SSL configuration).

---

## Troubleshooting

### Issue: Can't SSH into instance

**Solution:**
- Check security group allows SSH from your IP
- Verify key file permissions: `chmod 400 persona-rt-key.pem`
- Ensure using correct username: `ubuntu`
- Check instance is running in EC2 console

### Issue: Can't access application in browser

**Solution:**
- Check security group allows ports 80 and 8000
- Verify containers are running: `docker-compose ps`
- Check logs: `docker-compose logs`
- Test health endpoint: `curl http://localhost:8000/health`

### Issue: Frontend shows API errors

**Solution:**
- Verify `VITE_API_BASE_URL` in docker-compose.yml matches your EC2 IP
- Check backend is running: `docker-compose logs backend`
- Test API directly: `curl http://<YOUR_IP>:8000/health`

### Issue: Database errors

**Solution:**
- Check `data/` directory exists and has correct permissions
- View database file: `ls -la ~/persona-red-teaming/data/`
- Check backend logs: `docker-compose logs backend | grep -i database`

### Issue: Out of memory

**Solution:**
- Upgrade to larger instance type (t2.small, t2.medium)
- Monitor with: `free -h` and `docker stats`
- Consider adding swap space

### Issue: Docker build fails

**Solution:**
- Check disk space: `df -h`
- Clear Docker cache: `docker system prune -a`
- Pull latest code: `git pull`
- Rebuild: `docker-compose build --no-cache`

---

## Cost Estimation

### Free Tier (First 12 Months)
- **EC2 t2.micro**: 750 hours/month - FREE
- **EBS Storage**: 30 GB - FREE
- **Data Transfer**: 15 GB outbound - FREE

**Total for basic usage**: $0/month

### After Free Tier
- **EC2 t2.micro**: ~$8-10/month
- **EBS Storage (20 GB)**: ~$2/month
- **Data Transfer**: Minimal

**Total**: ~$10-12/month

### Cost Optimization Tips
- Stop instance when not in use (only pay for storage)
- Use reserved instances for 40-60% discount
- Monitor with AWS Cost Explorer
- Set up billing alerts

---

## Quick Reference Commands

```bash
# SSH into server
ssh -i ~/Downloads/persona-rt-key.pem ubuntu@<YOUR_IP>

# Navigate to app directory
cd ~/persona-red-teaming

# View status
docker-compose ps

# View logs
docker-compose logs -f

# Restart application
docker-compose restart

# Stop application
docker-compose down

# Start application
docker-compose up -d

# Update application
git pull && docker-compose up -d --build

# Backup database
cp data/history.db ~/backups/history_$(date +%Y%m%d).db

# Access database
sqlite3 data/history.db

# Check disk space
df -h

# Check memory
free -h
```

---

## Next Steps

1. ✅ Deploy application on EC2
2. Set up automated database backups
3. Configure domain name (optional)
4. Set up SSL/HTTPS (optional)
5. Monitor costs in AWS Billing dashboard
6. Set up CloudWatch alarms for instance health

---

## Support

For issues:
- Application bugs: https://github.com/WesDeng/persona-red-teaming/issues
- AWS EC2 help: https://docs.aws.amazon.com/ec2/
- Docker issues: https://docs.docker.com/

---

## Summary

You now have:
- ✅ Fully functional Persona Red Teaming application on AWS
- ✅ Persistent SQLite database with full access
- ✅ Direct SSH access to server
- ✅ Complete control over environment
- ✅ Free tier eligible deployment
- ✅ Easy database backup and export
- ✅ Ability to collect and analyze all user data

Your application is accessible at:
- **Frontend**: `http://<YOUR_EC2_PUBLIC_IP>`
- **Backend API**: `http://<YOUR_EC2_PUBLIC_IP>:8000`
- **Database**: SSH access at `~/persona-red-teaming/data/history.db`
