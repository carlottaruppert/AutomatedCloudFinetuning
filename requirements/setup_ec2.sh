#!/bin/bash
# EC2 Python 3.10 setup for TensorFlow + project dependencies
# Works on fresh Ubuntu 24.04 EC2 instances

set -e  # Exit on error

echo "=========================================="
echo "EC2 TRAINING ENVIRONMENT SETUP"
echo "=========================================="

# Check if we're on Ubuntu
if [ ! -f /etc/os-release ]; then
    echo "❌ Cannot detect OS version"
    exit 1
fi

source /etc/os-release
echo "Detected OS: $NAME $VERSION"

# Add deadsnakes PPA for Python 3.10
echo ""
echo "[1/6] Adding deadsnakes PPA..."
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.10
echo ""
echo "[2/6] Installing Python 3.10..."
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

# Verify Python 3.10 installation
echo ""
echo "[3/6] Verifying Python 3.10..."
python3.10 --version

# Remove old virtual environment if it exists and create new one
echo ""
echo "[4/6] Creating virtual environment..."
cd ~/training
if [ -d "training_env" ]; then
    echo "  Removing existing training_env..."
    rm -rf training_env
fi

python3.10 -m venv training_env

# Activate environment
source training_env/bin/activate

# Upgrade pip in the venv
echo ""
echo "[5/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo ""
echo "[6/6] Installing Python packages..."
echo "  This may take 5-10 minutes..."

# Install main requirements
pip install -r requirements/requirements_ec2.txt

# Install b-box-utils
pip install requirements/b_box_utils-1.4.6-py3-none-any.whl

# Verify installation
echo ""
echo "=========================================="
echo "VERIFICATION"
echo "=========================================="
python --version
python -c "import sys; print(f'Python: {sys.executable}')"
python -c "import tensorflow as tf; print(f'✅ TensorFlow: {tf.__version__}')"
python -c "from b_box_utils import PreprocessorMarginRight; print('✅ b-box-utils: OK')"
python -c "import numpy as np; print(f'✅ NumPy: {np.__version__}')"
python -c "import pandas as pd; print(f'✅ Pandas: {pd.__version__}')"
python -c "import matplotlib; print(f'✅ Matplotlib: {matplotlib.__version__}')"
python -c "import boto3; print(f'✅ Boto3: {boto3.__version__}')"

echo ""
echo "=========================================="
echo "✅ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Environment Summary:"
echo "  - Python 3.10.x"
echo "  - TensorFlow 2.16.2 (with CUDA support)"
echo "  - b-box-utils 1.4.6"
echo "  - All dependencies installed"
echo ""
echo "Next Steps:"
echo "  1. Activate environment:"
echo "     cd ~/training && source training_env/bin/activate"
echo ""
echo "  2. Start training:"
echo "     nohup python train_and_evaluate_on_ec2.py \\"
echo "       --s3-bucket YOUR_BUCKET_NAME \\"
echo "       2>&1 | tee training.log &"
echo ""
echo "  3. Monitor training:"
echo "     tail -f training.log"
echo ""
