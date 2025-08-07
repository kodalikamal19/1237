#!/bin/bash

# HackRX Optimization Application Script
# This script applies all optimizations to your existing HackRX project

set -e  # Exit on any error

echo "🚀 Starting HackRX Optimization Application..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "This script must be run from the root of your git repository"
    exit 1
fi

print_info "Detected git repository. Proceeding with optimizations..."

# Create backup branch
BACKUP_BRANCH="backup-before-hackrx-optimization-$(date +%Y%m%d_%H%M%S)"
git checkout -b "$BACKUP_BRANCH"
git checkout -

print_status "Created backup branch: $BACKUP_BRANCH"

# Check if required directories exist
if [ ! -d "src" ]; then
    print_error "src/ directory not found. Please run this script from your project root."
    exit 1
fi

# Create required directories if they don't exist
mkdir -p src/training
mkdir -p src/utils
mkdir -p src/routes

print_status "Directory structure verified"

# Step 1: Install new dependencies
print_info "Step 1: Updating requirements.txt..."

cat > requirements.txt << 'EOF'
annotated-types==0.7.0
anyio==4.9.0
blinker==1.9.0
cachetools==5.5.2
certifi==2025.8.3
charset-normalizer==3.4.2
click==8.2.1
distro==1.9.0
Flask==3.1.1
flask-cors==6.0.0
Flask-SQLAlchemy==3.1.1
google-ai-generativelanguage==0.6.15
google-api-core==2.25.1
google-api-python-client==2.177.0
google-auth==2.40.3
google-auth-httplib2==0.2.0
google-generativeai==0.8.5
googleapis-common-protos==1.70.0
greenlet==3.2.3
grpcio==1.74.0
grpcio-status==1.71.2
h11==0.16.0
httpcore==1.0.9
httplib2==0.22.0
httpx==0.28.1
idna==3.10
itsdangerous==2.2.0
Jinja2==3.1.6
jiter==0.10.0
MarkupSafe==3.0.2
proto-plus==1.26.1
protobuf==5.29.5
psutil==7.0.0
pyasn1==0.6.1
pyasn1_modules==0.4.2
pydantic==2.11.7
pydantic_core==2.33.2
pyparsing==3.2.3
pypdf==5.9.0
python-dotenv
requests==2.32.4
rsa==4.9.1
sniffio==1.3.1
SQLAlchemy==2.0.41
tqdm==4.67.1
typing-inspection==0.4.1
typing_extensions==4.14.0
uritemplate==4.2.0
urllib3==2.5.0
Werkzeug==3.1.3

scikit-learn==1.7.1
numpy==2.2.1
joblib==1.5.1
scipy==1.16.1
threadpoolctl==3.6.0

# OCR dependencies for image-based PDFs
pdf2image==1.17.0
pytesseract==0.3.13
Pillow==11.1.0

gunicorn==22.0.0

# Enhanced vector search and embeddings
faiss-cpu==1.8.0
sentence-transformers==3.3.1
transformers==4.47.1
torch==2.5.1

# Advanced text processing
spacy==3.8.3
nltk==3.9.1

# Performance optimization
redis==5.2.1
lru-dict==1.3.0

# Enhanced document processing
docx2txt==0.8
python-docx==1.1.2
EOF

print_status "Updated requirements.txt with new dependencies"

# Step 2: Create environment configuration
print_info "Step 2: Creating environment configuration..."

cat > .env.example << 'EOF'
# HackRX API Configuration
# Copy this file to .env and update with your actual values

# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Flask Configuration
SECRET_KEY=your_secret_key_here
FLASK_ENV=production
PORT=8000

# Cache Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_DIR=/tmp/hackrx_cache

# Memory Management
MAX_MEMORY_MB=2048
CACHE_SIZE=1000

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/hackrx.log

# Performance Tuning
MAX_WORKERS=4
TIMEOUT_SECONDS=120
MAX_DOCUMENT_SIZE_MB=100

# FAISS Configuration
FAISS_INDEX_TYPE=IndexFlatIP
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# HackRX Specific
HACKRX_MODE=production
OPTIMIZE_FOR_UNKNOWN_DOCS=true
ENABLE_SCORING_OPTIMIZER=true
ENABLE_CACHING=true
ENABLE_OCR=true

# Database (if needed for future features)
DATABASE_URL=postgresql://user:password@localhost:5432/hackrx

# Monitoring (optional)
SENTRY_DSN=your_sentry_dsn_here
ENABLE_METRICS=false
EOF

print_status "Created .env.example with HackRX optimizations"

# Update existing .env file if it exists
if [ -f ".env" ]; then
    print_info "Updating existing .env file with new variables..."
    
    # Add new variables if they don't exist
    grep -q "OPTIMIZE_FOR_UNKNOWN_DOCS" .env || echo "OPTIMIZE_FOR_UNKNOWN_DOCS=true" >> .env
    grep -q "ENABLE_SCORING_OPTIMIZER" .env || echo "ENABLE_SCORING_OPTIMIZER=true" >> .env
    grep -q "ENABLE_CACHING" .env || echo "ENABLE_CACHING=true" >> .env
    grep -q "ENABLE_OCR" .env || echo "ENABLE_OCR=true" >> .env
    grep -q "MAX_MEMORY_MB" .env || echo "MAX_MEMORY_MB=2048" >> .env
    grep -q "CACHE_SIZE" .env || echo "CACHE_SIZE=1000" >> .env
    grep -q "REDIS_URL" .env || echo "REDIS_URL=redis://localhost:6379/0" >> .env
    grep -q "CACHE_DIR" .env || echo "CACHE_DIR=/tmp/hackrx_cache" >> .env
    
    print_status "Updated existing .env file"
else
    print_warning ".env file not found. Please copy .env.example to .env and configure your API key"
fi

# Step 3: Download and create optimized files
print_info "Step 3: Creating optimized source files..."

# Check if we can download the optimized files from the zip
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -q -O"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -s -o"
else
    print_error "Neither wget nor curl found. Please install one of them to download files."
    print_info "Alternative: Extract files manually from hackrx-optimized-complete.zip"
    exit 1
fi

print_info "Creating new optimization modules..."

# Create the new files with placeholder content that needs to be replaced
echo "# This file needs to be replaced with optimized version from hackrx-optimized-complete.zip" > src/training/faiss_semantic_search.py
echo "# This file needs to be replaced with optimized version from hackrx-optimized-complete.zip" > src/training/optimized_query_processor.py
echo "# This file needs to be replaced with optimized version from hackrx-optimized-complete.zip" > src/training/scoring_optimizer.py
echo "# This file needs to be replaced with optimized version from hackrx-optimized-complete.zip" > src/training/enhanced_document_processor.py
echo "# This file needs to be replaced with optimized version from hackrx-optimized-complete.zip" > src/utils/intelligent_cache.py

print_status "Created placeholder files for optimization modules"

# Step 4: Create installation instructions
print_info "Step 4: Creating installation instructions..."

cat > HACKRX_INSTALLATION.md << 'EOF'
# HackRX Optimization Installation

## 🚀 Next Steps Required

This script has prepared your repository for HackRX optimizations. Complete the installation:

### 1. Replace Placeholder Files

Extract these files from `hackrx-optimized-complete.zip` and replace the placeholders:

- `src/training/faiss_semantic_search.py`
- `src/training/optimized_query_processor.py` 
- `src/training/scoring_optimizer.py`
- `src/training/enhanced_document_processor.py`
- `src/utils/intelligent_cache.py`
- `src/routes/hackrx_unified.py` (update existing)

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
sudo apt-get install redis-server  # Optional but recommended

# Install spaCy model
python -m spacy download en_core_web_sm
```

### 3. Configure Environment

1. Copy `.env.example` to `.env` if not exists: `cp .env.example .env`
2. Update `.env` with your Google API key
3. Configure other settings as needed

### 4. Test Installation

```bash
# Start the server
python src/main.py

# Test health endpoint
curl http://localhost:8000/api/v1/hackrx/health

# Should show new features like:
# - HackRX Scoring Optimization
# - FAISS semantic search  
# - Multi-layer intelligent caching
# - scoring_optimizer_available: true
```

## 🎯 HackRX Competition Features

✅ Unknown document prioritization (2.0x weight)
✅ FAISS semantic search
✅ Multi-layer intelligent caching  
✅ Advanced document processing
✅ Question complexity analysis
✅ Performance optimization

## 📞 Need Help?

Refer to the complete documentation in the zip file:
- `README.md` - Comprehensive guide
- `deployment_guide.md` - Production deployment
- `OPTIMIZATION_SUMMARY.md` - Technical details
EOF

print_status "Created installation instructions"

# Step 5: Commit changes
print_info "Step 5: Committing optimization preparations..."

git add .
git commit -m "Prepare for HackRX optimizations: updated dependencies and config

- Updated requirements.txt with FAISS, transformers, caching libraries
- Added .env.example with HackRX optimization settings
- Updated existing .env with new configuration variables
- Created placeholder files for optimization modules
- Added installation instructions

Next: Replace placeholder files with optimized versions from hackrx-optimized-complete.zip"

print_status "Committed optimization preparations"

# Final instructions
echo ""
echo "================================================"
print_status "HackRX Optimization Preparation Complete!"
echo "================================================"
echo ""
print_info "What was done:"
echo "  ✅ Created backup branch: $BACKUP_BRANCH"
echo "  ✅ Updated requirements.txt with new dependencies"
echo "  ✅ Created/updated environment configuration"
echo "  ✅ Created placeholder files for optimization modules"
echo "  ✅ Committed changes to git"
echo ""
print_warning "NEXT STEPS REQUIRED:"
echo "  1. Extract hackrx-optimized-complete.zip"
echo "  2. Replace placeholder files with optimized versions"
echo "  3. Install dependencies: pip install -r requirements.txt"
echo "  4. Install system dependencies (tesseract, redis)"
echo "  5. Configure your .env file with API key"
echo "  6. Test the installation"
echo ""
print_info "See HACKRX_INSTALLATION.md for detailed instructions"
echo ""
print_status "Ready for HackRX competition! 🏆"
EOF
