# HackRX Deployment Guide

## 🚀 Production Deployment Guide

This guide covers deploying the optimized HackRX system for maximum performance and reliability.

## 📋 Pre-Deployment Checklist

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 20GB+ SSD recommended
- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Python**: 3.8+ (3.11 recommended)

### Required Services
- **Redis**: For distributed caching (optional but recommended)
- **Nginx**: For reverse proxy and load balancing
- **Supervisor/systemd**: For process management
- **Monitoring**: Prometheus/Grafana (optional)

## 🔧 Installation Steps

### 1. System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    tesseract-ocr tesseract-ocr-eng \
    poppler-utils \
    redis-server \
    nginx \
    supervisor \
    git \
    build-essential

# CentOS/RHEL
sudo yum update -y
sudo yum install -y \
    python3.11 python3.11-devel \
    tesseract tesseract-langpack-eng \
    poppler-utils \
    redis \
    nginx \
    supervisor \
    git \
    gcc gcc-c++ make

# Enable and start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

### 2. Application Setup

```bash
# Create application user
sudo useradd -m -s /bin/bash hackrx
sudo usermod -aG sudo hackrx

# Switch to application user
sudo su - hackrx

# Clone repository
git clone <repository-url> /home/hackrx/hackrx-api
cd /home/hackrx/hackrx-api

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

### 3. Environment Configuration

```bash
# Create production environment file
cp .env.example .env

# Edit configuration
nano .env
```

**Production .env configuration:**
```bash
# Core Configuration
GOOGLE_API_KEY=your_actual_google_api_key
SECRET_KEY=your_secure_random_secret_key
FLASK_ENV=production
PORT=8000

# Cache Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_DIR=/home/hackrx/cache
CACHE_SIZE=2000

# Performance Configuration
MAX_MEMORY_MB=4096
MAX_WORKERS=4
TIMEOUT_SECONDS=120
MAX_DOCUMENT_SIZE_MB=100

# HackRX Optimizations
HACKRX_MODE=production
OPTIMIZE_FOR_UNKNOWN_DOCS=true
ENABLE_SCORING_OPTIMIZER=true
ENABLE_CACHING=true
ENABLE_OCR=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/hackrx/app.log

# FAISS Configuration
FAISS_INDEX_TYPE=IndexFlatIP
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

### 4. Directory Setup

```bash
# Create necessary directories
sudo mkdir -p /var/log/hackrx
sudo mkdir -p /home/hackrx/cache
sudo chown -R hackrx:hackrx /var/log/hackrx
sudo chown -R hackrx:hackrx /home/hackrx/cache

# Set permissions
chmod 755 /home/hackrx/hackrx-api
chmod 644 /home/hackrx/hackrx-api/.env
```

## 🔄 Process Management

### 1. Gunicorn Configuration

Create `/home/hackrx/hackrx-api/gunicorn.conf.py`:

```python
# Gunicorn configuration
bind = "127.0.0.1:8000"
workers = 4
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
timeout = 120
keepalive = 5
user = "hackrx"
group = "hackrx"

# Logging
accesslog = "/var/log/hackrx/access.log"
errorlog = "/var/log/hackrx/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "hackrx-api"

# Worker recycling
max_requests = 1000
max_requests_jitter = 100

# Memory management
worker_tmp_dir = "/dev/shm"
```

### 2. Supervisor Configuration

Create `/etc/supervisor/conf.d/hackrx.conf`:

```ini
[program:hackrx-api]
command=/home/hackrx/hackrx-api/venv/bin/gunicorn -c gunicorn.conf.py src.main:app
directory=/home/hackrx/hackrx-api
user=hackrx
group=hackrx
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/hackrx/supervisor.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
environment=PATH="/home/hackrx/hackrx-api/venv/bin"

[program:hackrx-cache-cleanup]
command=/home/hackrx/hackrx-api/venv/bin/python -c "from src.utils.intelligent_cache import get_cache_instance; get_cache_instance().cleanup_expired()"
directory=/home/hackrx/hackrx-api
user=hackrx
autostart=false
autorestart=false
environment=PATH="/home/hackrx/hackrx-api/venv/bin"
```

### 3. Start Services

```bash
# Update supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update

# Start application
sudo supervisorctl start hackrx-api

# Check status
sudo supervisorctl status hackrx-api
```

## 🌐 Nginx Configuration

### 1. Create Nginx Configuration

Create `/etc/nginx/sites-available/hackrx`:

```nginx
upstream hackrx_backend {
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=health:10m rate=1r/s;
    
    # Main API location
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://hackrx_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
        
        # Large request support
        client_max_body_size 100M;
        client_body_timeout 120s;
    }
    
    # Health check endpoint
    location /api/v1/hackrx/health {
        limit_req zone=health burst=5 nodelay;
        proxy_pass http://hackrx_backend;
        access_log off;
    }
    
    # Static files (if any)
    location /static/ {
        alias /home/hackrx/hackrx-api/src/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Deny access to sensitive files
    location ~ /\. {
        deny all;
    }
    
    location ~ \.(env|log|conf)$ {
        deny all;
    }
}
```

### 2. Enable Site

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/hackrx /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

## 📊 Monitoring & Logging

### 1. Log Rotation

Create `/etc/logrotate.d/hackrx`:

```
/var/log/hackrx/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 hackrx hackrx
    postrotate
        supervisorctl restart hackrx-api
    endscript
}
```

### 2. Health Monitoring Script

Create `/home/hackrx/scripts/health_check.sh`:

```bash
#!/bin/bash

# Health check script
HEALTH_URL="http://localhost:8000/api/v1/hackrx/health"
LOG_FILE="/var/log/hackrx/health_check.log"

# Check API health
response=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" --max-time 30)

timestamp=$(date '+%Y-%m-%d %H:%M:%S')

if [ "$response" = "200" ]; then
    echo "[$timestamp] HEALTHY: API responding normally" >> "$LOG_FILE"
    exit 0
else
    echo "[$timestamp] UNHEALTHY: API returned $response" >> "$LOG_FILE"
    # Restart service if unhealthy
    sudo supervisorctl restart hackrx-api
    exit 1
fi
```

### 3. Cron Jobs

Add to crontab (`crontab -e`):

```bash
# Health check every 5 minutes
*/5 * * * * /home/hackrx/scripts/health_check.sh

# Cache cleanup daily at 2 AM
0 2 * * * cd /home/hackrx/hackrx-api && /home/hackrx/hackrx-api/venv/bin/python -c "from src.utils.intelligent_cache import get_cache_instance; get_cache_instance().cleanup_expired()"

# Log cleanup weekly
0 3 * * 0 find /var/log/hackrx -name "*.log" -mtime +7 -delete
```

## 🔒 Security Configuration

### 1. Firewall Setup

```bash
# UFW firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8000/tcp  # Block direct access to app server
```

### 2. SSL/TLS Configuration (Let's Encrypt)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 3. Environment Security

```bash
# Secure environment file
chmod 600 /home/hackrx/hackrx-api/.env
chown hackrx:hackrx /home/hackrx/hackrx-api/.env

# Secure API key
# Use environment variables or secure key management
export GOOGLE_API_KEY="$(cat /home/hackrx/secrets/google_api_key)"
```

## 🚀 Performance Optimization

### 1. System Optimization

```bash
# Increase file limits
echo "hackrx soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "hackrx hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize Redis
sudo nano /etc/redis/redis.conf
# Add/modify:
# maxmemory 1gb
# maxmemory-policy allkeys-lru
# save 900 1
# save 300 10
# save 60 10000

sudo systemctl restart redis-server
```

### 2. Application Optimization

```bash
# Pre-warm cache on startup
cd /home/hackrx/hackrx-api
source venv/bin/activate
python -c "
from src.utils.intelligent_cache import get_cache_instance
cache = get_cache_instance()
print('Cache initialized and ready')
"
```

## 🔧 Troubleshooting

### Common Issues

1. **Service Won't Start**
   ```bash
   sudo supervisorctl tail hackrx-api stderr
   sudo journalctl -u supervisor -f
   ```

2. **High Memory Usage**
   ```bash
   # Monitor memory
   top -p $(pgrep -f hackrx-api)
   
   # Adjust worker count
   nano /home/hackrx/hackrx-api/gunicorn.conf.py
   # Reduce workers = 2
   ```

3. **Cache Issues**
   ```bash
   # Check Redis
   redis-cli ping
   redis-cli info memory
   
   # Clear cache
   redis-cli flushall
   ```

4. **API Timeouts**
   ```bash
   # Check Nginx logs
   sudo tail -f /var/log/nginx/error.log
   
   # Increase timeouts in nginx config
   proxy_read_timeout 300s;
   ```

### Performance Monitoring

```bash
# Monitor API performance
curl -w "@curl-format.txt" -s -o /dev/null http://localhost/api/v1/hackrx/health

# Monitor system resources
htop
iostat -x 1
free -h
df -h
```

### Backup Strategy

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/home/hackrx/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
tar -czf "$BACKUP_DIR/hackrx-$DATE.tar.gz" \
    /home/hackrx/hackrx-api \
    /var/log/hackrx \
    /etc/nginx/sites-available/hackrx \
    /etc/supervisor/conf.d/hackrx.conf

# Keep only last 7 days
find "$BACKUP_DIR" -name "hackrx-*.tar.gz" -mtime +7 -delete
```

## 📈 Scaling Considerations

### Horizontal Scaling

1. **Load Balancer Configuration**
   ```nginx
   upstream hackrx_backend {
       server 127.0.0.1:8000 weight=1;
       server 127.0.0.1:8001 weight=1;
       server 127.0.0.1:8002 weight=1;
   }
   ```

2. **Shared Cache**
   - Use Redis Cluster for distributed caching
   - Implement cache warming strategies
   - Monitor cache hit rates

3. **Database Scaling**
   - Consider PostgreSQL for persistent data
   - Implement read replicas
   - Use connection pooling

### Monitoring Dashboard

Set up Grafana dashboard for:
- Response times
- Memory usage
- Cache hit rates
- Error rates
- API throughput
- System metrics

## 🎯 HackRX Specific Optimizations

### Competition Mode Settings

```bash
# Environment optimizations for competition
export HACKRX_MODE=competition
export OPTIMIZE_FOR_UNKNOWN_DOCS=true
export ENABLE_SCORING_OPTIMIZER=true
export MAX_MEMORY_MB=8192
export CACHE_SIZE=5000
export MAX_WORKERS=6
```

### Pre-Competition Checklist

1. ✅ API key configured and tested
2. ✅ All dependencies installed
3. ✅ Cache system operational
4. ✅ OCR functionality verified
5. ✅ Memory limits appropriate
6. ✅ Logging configured
7. ✅ Health checks passing
8. ✅ Performance benchmarks met
9. ✅ Error handling tested
10. ✅ Backup strategy in place

---

**🏆 Ready for HackRX Competition!**

This deployment configuration is optimized for maximum performance and reliability during the HackRX competition, with focus on unknown document accuracy and scoring optimization.
