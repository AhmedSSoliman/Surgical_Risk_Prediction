#!/bin/bash

################################################################################
# Google Drive + GitHub Integration Setup
# This script sets up seamless synchronization between:
# - Local development (MacBook)
# - Google Drive (cloud storage + Colab access)
# - GitHub (version control)
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "🚀 SURGICAL RISK PREDICTION - GOOGLE DRIVE + GITHUB SETUP"
echo "================================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
PROJECT_DIR="/Users/ahmed.soliman/Documents/surgical_risk_prediction"
MIMIC_DIR="/Users/ahmed.soliman/Documents/surgical_risk_prediction/mimic-iii-clinical-database-1.4"

# Check for Google Drive installation
echo -e "\n${BLUE}[1/6] Checking Google Drive installation...${NC}"

GDRIVE_PATHS=(
    "$HOME/Library/CloudStorage/GoogleDrive-ahmed.soliman@ufl.edu/My Drive"
    "$HOME/Library/CloudStorage/GoogleDrive/My Drive"
    "$HOME/Google Drive/My Drive"
    "$HOME/GoogleDrive/My Drive"
)

GDRIVE_ROOT=""
for path in "${GDRIVE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        GDRIVE_ROOT="$path"
        echo -e "${GREEN}✅ Found Google Drive at: $GDRIVE_ROOT${NC}"
        break
    fi
done

if [ -z "$GDRIVE_ROOT" ]; then
    echo -e "${YELLOW}⚠️  Google Drive for Desktop not found${NC}"
    echo ""
    echo "Please install Google Drive for Desktop:"
    echo "  1. Visit: https://www.google.com/drive/download/"
    echo "  2. Download 'Drive for desktop' (macOS version)"
    echo "  3. Install and sign in with your Google account"
    echo "  4. Run this script again"
    echo ""
    echo "Alternative: Use the manual backup folder"
    GDRIVE_ROOT="$HOME/Documents/GoogleDrive_Backup"
    echo -e "${BLUE}Using local backup: $GDRIVE_ROOT${NC}"
fi

# Create project directory in Google Drive
echo -e "\n${BLUE}[2/6] Creating Google Drive project structure...${NC}"

GDRIVE_PROJECT="$GDRIVE_ROOT/SurgicalRiskPrediction"
mkdir -p "$GDRIVE_PROJECT"
mkdir -p "$GDRIVE_PROJECT/models"
mkdir -p "$GDRIVE_PROJECT/data/processed"
mkdir -p "$GDRIVE_PROJECT/results"
mkdir -p "$GDRIVE_PROJECT/figures"
mkdir -p "$GDRIVE_PROJECT/notebooks"

echo -e "${GREEN}✅ Created: $GDRIVE_PROJECT${NC}"

# Copy project files (excluding large files and git)
echo -e "\n${BLUE}[3/6] Copying project files to Google Drive...${NC}"
echo "   (Excluding: .git, models/checkpoints, large data files)"

# Use rsync for efficient copying
rsync -av --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints' \
    --exclude='models/checkpoints/*.pt' \
    --exclude='mimic-iii-clinical-database-1.4' \
    --exclude='data/raw' \
    --exclude='data/processed/*.pkl' \
    --exclude='results/*.csv' \
    --exclude='*.log' \
    "$PROJECT_DIR/" "$GDRIVE_PROJECT/" 2>&1 | tail -20

echo -e "${GREEN}✅ Project files copied${NC}"

# Handle MIMIC-III database
echo -e "\n${BLUE}[4/6] Handling MIMIC-III database...${NC}"
echo "   Size: ~50GB (too large for Google Drive free tier)"
echo ""
echo "Options for MIMIC-III:"
echo "   A) Keep local only (recommended for development)"
echo "   B) Create symlink in Google Drive (points to local file)"
echo "   C) Upload to Google Drive (requires 100GB+ storage)"
echo "   D) Use Colab with mounted Drive (download on-demand)"
echo ""

# Create a metadata file about MIMIC location
cat > "$GDRIVE_PROJECT/MIMIC_DATABASE_INFO.txt" << EOF
================================================================================
MIMIC-III CLINICAL DATABASE INFORMATION
================================================================================

Local Location: $MIMIC_DIR
Size: ~50GB
Files: CSV files with clinical data

IMPORTANT: Due to size, MIMIC-III is NOT synced to Google Drive by default.

Options to access in Google Colab:
-----------------------------------

Option 1: Download from PhysioNet (Recommended for Colab)
  1. Visit: https://physionet.org/content/mimiciii/1.4/
  2. Complete required training
  3. Download directly to Colab:
     !wget -r -N -c -np --user YOUR_USERNAME --ask-password \\
          https://physionet.org/files/mimiciii/1.4/

Option 2: Upload compressed version to Google Drive
  1. Compress locally: tar -czf mimic3.tar.gz mimic-iii-clinical-database-1.4/
  2. Upload to Google Drive (takes ~6 hours, needs 100GB+ storage)
  3. Extract in Colab: !tar -xzf /content/drive/MyDrive/mimic3.tar.gz

Option 3: Use sample data for testing
  - Project includes sample data in data/sample/
  - Good for testing code without full database

Current Setup: Local only
Update this file if you change the storage strategy.
================================================================================
EOF

echo -e "${GREEN}✅ Created MIMIC database info file${NC}"

# Copy model checkpoints
echo -e "\n${BLUE}[5/6] Copying trained models...${NC}"

if [ -d "$PROJECT_DIR/models/checkpoints" ]; then
    rsync -av --progress \
        "$PROJECT_DIR/models/checkpoints/" \
        "$GDRIVE_PROJECT/models/checkpoints/" 2>&1 | tail -10
    
    SIZE=$(du -sh "$GDRIVE_PROJECT/models/checkpoints" | cut -f1)
    echo -e "${GREEN}✅ Models copied (${SIZE})${NC}"
else
    echo -e "${YELLOW}⚠️  No model checkpoints found yet${NC}"
fi

# Create sync configuration
echo -e "\n${BLUE}[6/6] Creating sync configuration...${NC}"

cat > "$GDRIVE_PROJECT/.sync_config" << EOF
# Google Drive + GitHub Sync Configuration
# Generated: $(date)

GITHUB_REPO=AhmedSSoliman/Surgical_Risk_Prediction
LOCAL_PATH=$PROJECT_DIR
GDRIVE_PATH=$GDRIVE_PROJECT
BRANCH=master

# Files to sync
SYNC_INCLUDE=*.py,*.ipynb,*.md,*.txt,*.yml,*.yaml,*.json,*.sh,requirements.txt

# Files to exclude
SYNC_EXCLUDE=*.pt,*.pth,*.pkl,*.csv,*.h5,__pycache__,*.pyc,.git,.ipynb_checkpoints

# Sync frequency
AUTO_SYNC=false  # Set to true for automatic sync
SYNC_INTERVAL=3600  # seconds (1 hour)
EOF

# Create helper scripts
cat > "$GDRIVE_PROJECT/sync_to_github.sh" << 'SYNCEOF'
#!/bin/bash
# Quick sync to GitHub from Google Drive

cd "$(dirname "$0")"
echo "🔄 Syncing to GitHub..."

git add .
git commit -m "Update from Google Drive: $(date '+%Y-%m-%d %H:%M:%S')"
git push origin master

echo "✅ Synced to GitHub!"
SYNCEOF

chmod +x "$GDRIVE_PROJECT/sync_to_github.sh"

cat > "$GDRIVE_PROJECT/sync_from_github.sh" << 'PULLEOF'
#!/bin/bash
# Quick sync from GitHub to Google Drive

cd "$(dirname "$0")"
echo "🔄 Syncing from GitHub..."

git pull origin master

echo "✅ Synced from GitHub!"
PULLEOF

chmod +x "$GDRIVE_PROJECT/sync_from_github.sh"

echo -e "${GREEN}✅ Sync configuration created${NC}"

# Create README for Google Drive folder
cat > "$GDRIVE_PROJECT/GOOGLE_DRIVE_README.md" << 'EOF'
# Surgical Risk Prediction - Google Drive Folder

This folder contains your complete Surgical Risk Prediction project, synced with Google Drive for:
- ☁️ Cloud backup
- 🔄 Access from Google Colab
- 👥 Collaboration
- 📱 Cross-device development

## Folder Structure

```
SurgicalRiskPrediction/
├── models/
│   ├── checkpoints/          # Trained model weights (synced)
│   ├── model.py
│   └── vibe_tuning.py
├── data/
│   ├── processed/            # Preprocessed data
│   └── sample/               # Sample data for testing
├── notebooks/                # Jupyter notebooks
├── training/                 # Training scripts
├── preprocessing/            # Data preprocessing
├── explainability/           # Model interpretation
├── results/                  # Experiment results
└── figures/                  # Visualizations

📝 MIMIC-III database (50GB) is stored locally only.
   See MIMIC_DATABASE_INFO.txt for details.
```

## Working with This Folder

### Option 1: Local Development (MacBook)
Your local project automatically syncs to this Google Drive folder.

```bash
cd /Users/ahmed.soliman/Documents/surgical_risk_prediction
# Make changes, they sync automatically
```

### Option 2: Google Colab Development

In Colab notebook:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
import os
os.chdir('/content/drive/MyDrive/SurgicalRiskPrediction')

# Now you can import modules and run code
from models.vibe_tuning import VibeTunedBiomedicalEncoder
```

### Option 3: Clone from GitHub

```bash
git clone https://github.com/AhmedSSoliman/Surgical_Risk_Prediction.git
cd Surgical_Risk_Prediction
```

## Sync Workflow

### Push changes to GitHub:
```bash
cd /path/to/SurgicalRiskPrediction
./sync_to_github.sh
```

### Pull latest from GitHub:
```bash
cd /path/to/SurgicalRiskPrediction
./sync_from_github.sh
```

### Manual Git commands:
```bash
git add .
git commit -m "Your message"
git push origin master
```

## Best Practices

1. **Regular Commits**: Commit changes frequently to GitHub
2. **Pull Before Push**: Always pull latest changes before pushing
3. **Model Versioning**: Keep multiple checkpoint versions
4. **Documentation**: Update README when adding features
5. **Testing**: Test in Colab before pushing major changes

## Troubleshooting

**Problem**: "Google Drive folder not syncing"
- Check Google Drive for Desktop is running
- Verify internet connection
- Check storage quota (need ~500MB free)

**Problem**: "Can't access in Colab"
- Ensure Drive is mounted: `drive.mount('/content/drive')`
- Check folder path is correct
- Verify file permissions

**Problem**: "Git conflicts"
- Pull latest changes: `git pull origin master`
- Resolve conflicts manually
- Commit and push

## Contact

For issues or questions about the project:
- GitHub: https://github.com/AhmedSSoliman/Surgical_Risk_Prediction
- Email: ahmed.soliman@ufl.edu
EOF

echo -e "${GREEN}✅ Created Google Drive README${NC}"

# Summary
echo ""
echo "================================================================================"
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo "================================================================================"
echo ""
echo "📁 Google Drive Location:"
echo "   $GDRIVE_PROJECT"
echo ""
echo "🔗 GitHub Repository:"
echo "   https://github.com/AhmedSSoliman/Surgical_Risk_Prediction"
echo ""
echo "📊 Summary:"
du -sh "$GDRIVE_PROJECT" 2>/dev/null | awk '{print "   Total size: " $1}'
find "$GDRIVE_PROJECT" -type f | wc -l | awk '{print "   Files: " $1}'
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1️⃣  Initialize Git in Google Drive folder:"
echo "   cd \"$GDRIVE_PROJECT\""
echo "   git init"
echo "   git remote add origin https://github.com/AhmedSSoliman/Surgical_Risk_Prediction.git"
echo "   git fetch origin"
echo "   git checkout master"
echo ""
echo "2️⃣  Or link to existing local repo:"
echo "   cd \"$GDRIVE_PROJECT\""
echo "   rm -rf .git"
echo "   ln -s \"$PROJECT_DIR/.git\" .git"
echo ""
echo "3️⃣  Test in Google Colab:"
echo "   - Open: https://colab.research.google.com"
echo "   - Upload: surgical_risk_prediction_notebook.ipynb"
echo "   - Mount Drive and run!"
echo ""
echo "================================================================================"
