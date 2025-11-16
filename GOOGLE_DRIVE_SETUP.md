# Google Drive + GitHub Integration Guide

This guide explains how to set up seamless synchronization between your local machine, Google Drive, and GitHub for the Surgical Risk Prediction project.

## 🎯 Goal

Create a unified workflow where you can:
- ✅ Work locally on your MacBook
- ✅ Auto-sync to Google Drive
- ✅ Access from Google Colab
- ✅ Version control with GitHub
- ✅ Share with collaborators

## 📋 Prerequisites

### 1. Install Google Drive for Desktop (Recommended)

**Why**: Automatic synchronization between local folder and Google Drive

**Installation**:
1. Visit: https://www.google.com/drive/download/
2. Download "Drive for desktop" for macOS
3. Install and sign in with your Google account (ahmed.soliman@ufl.edu)
4. Choose sync location (default: `~/Library/CloudStorage/GoogleDrive-ahmed.soliman@ufl.edu/My Drive`)

**Alternative**: Use the manual backup folder created by the setup script

### 2. Verify GitHub Access

```bash
# Check if you can push to GitHub
cd /Users/ahmed.soliman/Documents/surgical_risk_prediction
git remote -v
# Should show: origin https://github.com/AhmedSSoliman/Surgical_Risk_Prediction.git
```

## 🚀 Setup Methods

### Method 1: Automatic Setup (Recommended)

Run the provided setup script:

```bash
cd /Users/ahmed.soliman/Documents/surgical_risk_prediction
./setup_google_drive_sync.sh
```

This will:
- ✅ Detect or create Google Drive folder
- ✅ Copy project files (excluding large data)
- ✅ Set up sync configuration
- ✅ Create helper scripts
- ✅ Generate documentation

### Method 2: Manual Setup

#### Step 1: Create Google Drive Folder

```bash
# If Google Drive is installed
GDRIVE_ROOT="$HOME/Library/CloudStorage/GoogleDrive-ahmed.soliman@ufl.edu/My Drive"

# Or use backup location
GDRIVE_ROOT="$HOME/Documents/GoogleDrive_Backup"

# Create project folder
mkdir -p "$GDRIVE_ROOT/SurgicalRiskPrediction"
```

#### Step 2: Copy Project Files

```bash
# Copy everything except large files
rsync -av --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.ipynb_checkpoints' \
    --exclude='models/checkpoints/*.pt' \
    --exclude='mimic-iii-clinical-database-1.4' \
    --exclude='data/raw' \
    /Users/ahmed.soliman/Documents/surgical_risk_prediction/ \
    "$GDRIVE_ROOT/SurgicalRiskPrediction/"
```

#### Step 3: Link Git Repository

**Option A**: Clone fresh copy in Google Drive
```bash
cd "$GDRIVE_ROOT/SurgicalRiskPrediction"
git init
git remote add origin https://github.com/AhmedSSoliman/Surgical_Risk_Prediction.git
git fetch origin
git checkout master
```

**Option B**: Symlink to existing .git folder
```bash
cd "$GDRIVE_ROOT/SurgicalRiskPrediction"
ln -s /Users/ahmed.soliman/Documents/surgical_risk_prediction/.git .git
```

## 📁 Folder Structure

After setup, you'll have:

```
Google Drive/
└── SurgicalRiskPrediction/          # Synced to cloud
    ├── models/
    │   ├── checkpoints/             # Trained models (~350MB)
    │   ├── model.py
    │   └── vibe_tuning.py
    ├── data/
    │   ├── processed/               # Preprocessed data
    │   └── sample/                  # Sample data
    ├── preprocessing/               # Code files
    ├── training/
    ├── explainability/
    ├── utils/
    ├── notebooks/
    ├── requirements.txt
    ├── README.md
    ├── sync_to_github.sh           # Helper script
    ├── sync_from_github.sh         # Helper script
    └── MIMIC_DATABASE_INFO.txt     # Info about MIMIC-III

Local Only:
/Users/ahmed.soliman/Documents/surgical_risk_prediction/
└── mimic-iii-clinical-database-1.4/  # 50GB - too large for cloud
```

## 🔄 Workflow Options

### Workflow 1: Local Development → GitHub → Google Drive

**Best for**: Primary development on MacBook

```bash
# 1. Work locally
cd /Users/ahmed.soliman/Documents/surgical_risk_prediction
# Make changes...

# 2. Commit to GitHub
git add .
git commit -m "Updated vibe tuning"
git push origin master

# 3. Google Drive syncs automatically (if Drive for Desktop installed)
# Or manually copy:
rsync -av --exclude='.git' --exclude='mimic-iii*' \
    /Users/ahmed.soliman/Documents/surgical_risk_prediction/ \
    "$GDRIVE_ROOT/SurgicalRiskPrediction/"
```

### Workflow 2: Google Drive → GitHub (from Colab)

**Best for**: Experiments and training in Colab

In Google Colab:
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
import os
os.chdir('/content/drive/MyDrive/SurgicalRiskPrediction')

# Make changes...

# Commit to GitHub
!git add .
!git commit -m "Updated from Colab"
!git push origin master
```

### Workflow 3: Hybrid (Recommended)

1. **Development**: Work locally with full MIMIC-III data
2. **Sync**: Push to GitHub frequently
3. **Experiments**: Run in Colab with sample data or download MIMIC-III on-demand
4. **Results**: Models and results sync via Google Drive

## 🔧 Helper Scripts

### sync_to_github.sh

Quick push to GitHub:
```bash
cd /path/to/SurgicalRiskPrediction
./sync_to_github.sh
```

### sync_from_github.sh

Quick pull from GitHub:
```bash
cd /path/to/SurgicalRiskPrediction
./sync_from_github.sh
```

### Manual Git Commands

```bash
# Check status
git status

# Add all changes
git add .

# Commit with message
git commit -m "Your message here"

# Push to GitHub
git push origin master

# Pull latest changes
git pull origin master

# View commit history
git log --oneline -10
```

## 🗄️ Handling MIMIC-III Database (50GB)

The MIMIC-III database is too large for Google Drive free tier (15GB). Options:

### Option 1: Local Only (Current Setup)
- Keep MIMIC-III on MacBook only
- Use for local development
- Use sample data in Colab

### Option 2: Download in Colab On-Demand
```python
# In Colab, download from PhysioNet
!wget -r -N -c -np \
    --user YOUR_USERNAME --ask-password \
    https://physionet.org/files/mimiciii/1.4/
```

### Option 3: Upload Compressed to Google Drive
```bash
# Compress (takes ~1 hour, creates ~20GB file)
tar -czf mimic3.tar.gz mimic-iii-clinical-database-1.4/

# Upload to Google Drive manually
# Then in Colab:
!tar -xzf /content/drive/MyDrive/mimic3.tar.gz
```

### Option 4: Use Google Cloud Storage
- Upload to GCS bucket
- Mount in Colab: `!gcsfuse bucket-name /mnt/gcs`
- Faster and more reliable than Drive for large datasets

## 📊 Storage Requirements

| Location | Size | Purpose |
|----------|------|---------|
| **Local MacBook** | ~56GB | Full development + MIMIC-III |
| **Google Drive** | ~500MB | Code + models + sample data |
| **GitHub** | ~50MB | Code only (no models/data) |
| **Colab Runtime** | ~500MB-50GB | On-demand (ephemeral) |

## 🎓 Google Colab Usage

### Initial Setup in Colab

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Navigate to project
import os
import sys

project_path = '/content/drive/MyDrive/SurgicalRiskPrediction'
os.chdir(project_path)
sys.path.insert(0, project_path)

print(f"✅ Working directory: {os.getcwd()}")

# Cell 3: Install dependencies
!pip install -q -r requirements.txt

# Cell 4: Verify imports
from models.vibe_tuning import VibeTunedBiomedicalEncoder
from utils.utils import get_device

device = get_device()
print(f"✅ Device: {device}")
```

### Load Trained Model in Colab

```python
import torch
from pathlib import Path

# Load from Google Drive
model_path = Path('/content/drive/MyDrive/SurgicalRiskPrediction/models/checkpoints/best_model.pt')
checkpoint = torch.load(model_path, map_location=device)

# Initialize model
model = VibeTunedBiomedicalEncoder(
    base_model='emilyalsentzer/Bio_ClinicalBERT',
    use_lora=True,
    use_adapters=True
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✅ Model loaded! Epoch: {checkpoint['epoch']}, Loss: {checkpoint['best_val_loss']:.4f}")
```

## 🐛 Troubleshooting

### Problem: "Git push rejected"

**Solution**:
```bash
# Pull first, then push
git pull origin master
# Resolve conflicts if any
git push origin master
```

### Problem: "Google Drive not syncing"

**Solutions**:
1. Check Google Drive for Desktop is running (menu bar icon)
2. Check internet connection
3. Check storage quota: https://drive.google.com/drive/quota
4. Restart Google Drive: Quit and reopen
5. Check sync settings: Preferences → Google Drive → Sync folders

### Problem: "Can't import modules in Colab"

**Solution**:
```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/SurgicalRiskPrediction')
```

### Problem: "Permission denied" in Colab

**Solution**:
```bash
# In Colab notebook
!chmod +x /content/drive/MyDrive/SurgicalRiskPrediction/*.sh
```

### Problem: "Git detached HEAD"

**Solution**:
```bash
git checkout master
git pull origin master
```

## 📝 Best Practices

1. **Commit Frequently**: Small, focused commits with clear messages
2. **Pull Before Push**: Always pull latest changes first
3. **Use .gitignore**: Don't commit large files or sensitive data
4. **Test in Colab**: Verify notebook works in Colab before sharing
5. **Version Models**: Keep multiple checkpoint versions
6. **Document Changes**: Update README with new features
7. **Backup Regularly**: GitHub + Google Drive = double backup

## 🔐 Security Notes

- ✅ **Do commit**: Code, notebooks, configs, documentation
- ❌ **Don't commit**: API keys, passwords, PHI/patient data, large datasets
- ✅ **Use**: `.gitignore` to exclude sensitive files
- ✅ **Check**: Always review `git status` before committing

## 📚 Additional Resources

- **GitHub Repository**: https://github.com/AhmedSSoliman/Surgical_Risk_Prediction
- **Google Colab**: https://colab.research.google.com
- **Google Drive**: https://drive.google.com
- **MIMIC-III**: https://physionet.org/content/mimiciii/1.4/
- **Git Documentation**: https://git-scm.com/doc

## 💬 Support

For questions or issues:
- **GitHub Issues**: https://github.com/AhmedSSoliman/Surgical_Risk_Prediction/issues
- **Email**: ahmed.soliman@ufl.edu

---

**Last Updated**: November 16, 2025  
**Status**: ✅ Fully Configured and Ready to Use
