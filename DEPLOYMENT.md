# Deployment Guide

## Quick Start (Development)

```powershell
# Windows PowerShell
python api.py

# Or with uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# Linux/Mac
python api.py

# Or with uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Production Deployment

### Option 1: Direct Uvicorn (Simple)

```bash
# Linux/Mac
uvicorn api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info

# Windows PowerShell
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Option 2: Using Startup Scripts

**Linux/Mac:**
```bash
chmod +x start_api.sh
./start_api.sh
```

**Windows:**
```cmd
start_api.bat
```

### Option 3: Systemd Service (Linux Production)

1. **Edit the service file:**
   ```bash
   sudo nano /etc/systemd/system/geotechnical-api.service
   ```
   
   Update paths in `api.service`:
   - `WorkingDirectory`: Full path to your project
   - `ExecStart`: Full path to uvicorn in your venv
   - `User`: Your server user (or www-data)

2. **Enable and start:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable geotechnical-api
   sudo systemctl start geotechnical-api
   ```

3. **Check status:**
   ```bash
   sudo systemctl status geotechnical-api
   ```

4. **View logs:**
   ```bash
   sudo journalctl -u geotechnical-api -f
   ```

### Option 4: Docker (Recommended for Production)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and run:
```bash
docker build -t geotechnical-api .
docker run -d -p 8000:8000 --env-file .env geotechnical-api
```

### Option 5: Using Gunicorn + Uvicorn Workers

```bash
pip install gunicorn

gunicorn api:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

## Environment Setup

1. **Create `.env` file:**
   ```env
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   SUPABASE_STORAGE_BUCKET=geotechnical-data
   PORT=8000
   WORKERS=4
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Reverse Proxy (Nginx)

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Process Management

### Using PM2 (Node.js process manager)

```bash
npm install -g pm2

pm2 start api.py --name geotechnical-api --interpreter python3
pm2 save
pm2 startup
```

### Using Supervisor

```ini
[program:geotechnical-api]
command=/path/to/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
directory=/path/to/project
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/geotechnical-api.err.log
stdout_logfile=/var/log/geotechnical-api.out.log
```

## Health Check

Test the API:
```bash
curl http://localhost:8000/health
```

## Troubleshooting

1. **Port already in use:**
   ```bash
   # Find process using port 8000
   lsof -i :8000  # Linux/Mac
   netstat -ano | findstr :8000  # Windows
   
   # Kill process or change port
   export PORT=8001
   ```

2. **Model not loading:**
   - Check `.env` file has correct credentials
   - Verify models exist in Supabase Storage: `models/scaler.pkl`, `models/ann_liquefaction_classifier.pkl`
   - Check API logs for errors

3. **Database connection issues:**
   - Verify `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in `.env`
   - Test connection: `python -c "from supabase import create_client; import os; from dotenv import load_dotenv; load_dotenv(); client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_ROLE_KEY')); print('Connected!')"`

## Performance Tuning

- **Workers**: Set `--workers` to (2 × CPU cores) + 1
- **Timeout**: Increase `--timeout-keep-alive` for slow queries
- **Logging**: Use `--log-level warning` in production

## Security

1. **Use HTTPS** (via reverse proxy)
2. **Restrict CORS** origins in production
3. **Rate limiting** (add middleware)
4. **API keys** for authentication (optional)
