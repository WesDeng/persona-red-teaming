# Deployment Guide

This guide covers deploying the Persona Red-Teaming application to Railway, Render, or Docker.

## Prerequisites

- Docker and Docker Compose installed (for local Docker deployment)
- Railway or Render account (for cloud deployment)
- Git repository pushed to GitHub/GitLab

## Option 1: Local Docker Deployment

### Build and Run

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop services
docker-compose down
```

### Access the Application

- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

## Option 2: Railway Deployment

### Backend Deployment

1. **Create New Project on Railway**
   - Go to https://railway.app
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Configure Backend Service**
   - Railway will auto-detect the Dockerfile
   - Add environment variables:
     ```
     DATABASE_PATH=/data/history.db
     OPENAI_API_KEY=your-openai-key
     ```
   - Railway will auto-assign a public URL

3. **Add Persistent Volume** (Important!)
   - Go to service settings → "Volumes"
   - Add volume: `/data` (for SQLite database)

### Frontend Deployment

1. **Create Another Service**
   - In the same Railway project, add a new service
   - Select your repository again
   - Choose the frontend Dockerfile

2. **Configure Frontend**
   - Add environment variable:
     ```
     VITE_API_BASE_URL=https://your-backend-url.railway.app
     ```
   - Railway will provide a public URL

## Option 3: Render Deployment

### One-Click Deploy

1. **Connect Repository**
   - Go to https://render.com
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Render will read `render.yaml` and create both services

2. **Configure Environment Variables**
   - Backend: Set `OPENAI_API_KEY` in dashboard
   - Frontend: Set `VITE_API_BASE_URL` to your backend URL
     - Example: `https://persona-rt-api.onrender.com`

3. **Wait for Deployment**
   - Render will automatically build and deploy both services
   - Backend will have persistent disk for SQLite database

### Manual Deploy (Alternative)

#### Backend:
```bash
1. New Web Service → Docker
2. Build Command: (auto-detected from Dockerfile)
3. Environment Variables:
   - DATABASE_PATH=/data/history.db
   - OPENAI_API_KEY=your-key
4. Add Disk: /data with 1GB
```

#### Frontend:
```bash
1. New Web Service → Docker
2. Build Command: (auto-detected from Dockerfile)
3. Environment Variables:
   - VITE_API_BASE_URL=https://your-backend.onrender.com
```

## Environment Variables Reference

### Backend (Required)
| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLMs | `sk-...` |
| `DATABASE_PATH` | Path to SQLite database | `/data/history.db` |

### Frontend (Required)
| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | `https://api.example.com` |

## Post-Deployment Checklist

- [ ] Backend health check working: `GET /health`
- [ ] Frontend loads successfully
- [ ] Can create and view persona history
- [ ] Can view prompt history
- [ ] Database persists after restart
- [ ] CORS allows frontend origin
- [ ] API keys are set correctly

## Troubleshooting

### Database Not Persisting
- **Railway**: Ensure volume is mounted to `/data`
- **Render**: Ensure disk is attached to `/data`
- **Docker**: Check volume mapping in `docker-compose.yml`

### CORS Errors
- Check backend CORS settings in `api/server.py`
- Verify `VITE_API_BASE_URL` is set correctly in frontend

### Frontend Can't Connect to Backend
- Ensure `VITE_API_BASE_URL` points to correct backend URL
- Check backend is running and accessible
- Verify health endpoint: `curl https://your-backend/health`

### Build Failures
- **Frontend**: Check Node version (requires 18+)
- **Backend**: Check Python version (requires 3.11+)
- Check Docker logs: `docker-compose logs`

## Monitoring

### Railway
- Built-in metrics dashboard
- View logs in real-time
- Set up usage alerts

### Render
- Performance metrics available
- Log streaming in dashboard
- Email notifications for failures

### Docker (Local)
```bash
# Resource usage
docker stats

# Container health
docker ps

# Logs
docker-compose logs -f
```

## Scaling Considerations

For 10-15 users (current scope):
- **Single instance** is sufficient
- **SQLite** is adequate
- **No load balancer** needed

For scaling beyond 50+ users:
- Consider PostgreSQL instead of SQLite
- Add Redis for caching
- Use multiple backend replicas
- Add CDN for frontend assets

## Backup

### Backup Database (Important!)

```bash
# Railway/Render: Download via dashboard or SSH
# Docker local: Database is in ./data/history.db

# Backup command
cp data/history.db backups/history-$(date +%Y%m%d).db
```

### Automated Backups
- Railway: Use volume snapshots
- Render: Use disk snapshots
- Docker: Create cron job for backups

## Security Notes

1. **API Keys**: Never commit keys to git
2. **CORS**: Restrict origins in production if possible
3. **HTTPS**: Always use HTTPS in production
4. **Database**: Regular backups recommended
5. **Updates**: Keep dependencies updated

## Support

For issues, check:
1. Service logs
2. Health check endpoint
3. Database file exists and has write permissions
4. Environment variables are set correctly
