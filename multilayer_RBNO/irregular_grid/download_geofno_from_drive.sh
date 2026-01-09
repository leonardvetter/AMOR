#!/bin/bash
# Script to download Geo-FNO datasets from Google Drive directly to server
# Usage: ./download_geofno_from_drive.sh

set -e

# Set target directory
SCRATCH_DIR="${SCRATCH:-/pscratch/sd/d/dvod}"
DATA_DIR="${SCRATCH_DIR}/data/Geo-FNO/data"

# Create directories
mkdir -p "${DATA_DIR}/naca"
mkdir -p "${DATA_DIR}/diffeo"

echo "============================================================"
echo "Geo-FNO Dataset Downloader from Google Drive"
echo "============================================================"
echo ""

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing..."
    pip install gdown
fi

echo "To download files from Google Drive, you have two options:"
echo ""
echo "OPTION 1: Using gdown with file IDs (recommended)"
echo "---------------------------------------------------"
echo "1. Open the Google Drive folder:"
echo "   https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8?usp=sharing"
echo ""
echo "2. For each file you need, right-click -> 'Get link' -> Extract the FILE_ID"
echo "   Example: https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
echo ""
echo "3. Download individual files using:"
echo "   gdown --id FILE_ID -O output_file.npy"
echo ""
echo "For Airfoil dataset (data/naca/):"
echo "  cd ${DATA_DIR}/naca"
echo "  gdown --id <NACA_Cylinder_X_FILE_ID> -O NACA_Cylinder_X.npy"
echo "  gdown --id <NACA_Cylinder_Y_FILE_ID> -O NACA_Cylinder_Y.npy"
echo "  gdown --id <NACA_Cylinder_Q_FILE_ID> -O NACA_Cylinder_Q.npy"
echo ""
echo "For Elasticity dataset (data/diffeo/):"
echo "  cd ${DATA_DIR}/diffeo"
echo "  gdown --id <Random_UnitCell_rr_10_FILE_ID> -O Random_UnitCell_rr_10.npy"
echo "  gdown --id <Random_UnitCell_XY_10_FILE_ID> -O Random_UnitCell_XY_10.npy"
echo "  gdown --id <Random_UnitCell_sigma_10_FILE_ID> -O Random_UnitCell_sigma_10.npy"
echo ""
echo "OPTION 2: Using gdown with folder ID (if folder is shared)"
echo "------------------------------------------------------------"
echo "If the entire folder is shared, you can try:"
echo "  gdown --folder --id 1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8 -O ${DATA_DIR}/geofno_raw"
echo "  (Then move files to appropriate subdirectories)"
echo ""
echo "OPTION 3: Using wget/curl (for direct download links)"
echo "------------------------------------------------------"
echo "If you can get direct download links, use:"
echo "  wget -O output_file.npy 'DIRECT_DOWNLOAD_URL'"
echo "  # or"
echo "  curl -L -o output_file.npy 'DIRECT_DOWNLOAD_URL'"
echo ""
echo "To get direct download link from Google Drive file ID:"
echo "  https://drive.google.com/uc?export=download&id=FILE_ID"
echo ""
echo "Example:"
echo "  wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=FILE_ID' -O output_file.npy"
echo ""
echo "============================================================"
echo "After downloading, run:"
echo "  python download_geofno_datasets.py"
echo "to verify the datasets are in the correct location."
echo "============================================================"

