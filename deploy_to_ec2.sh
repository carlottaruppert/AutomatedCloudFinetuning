#!/bin/bash
# Master deployment script - transfers files to EC2 and runs setup

set -e

# Configuration
EC2_IP="${1}"
KEY_PATH="${2}"
EC2_USER="ubuntu"

# Check if required arguments are provided
if [ -z "$EC2_IP" ] || [ -z "$KEY_PATH" ]; then
    echo "Usage: $0 <EC2_IP> <KEY_PATH>"
    echo "Example: $0 35.159.64.90 ~/.ssh/your-keypair.pem"
    exit 1
fi

echo "=========================================="
echo "EC2 DEPLOYMENT"
echo "=========================================="
echo "EC2 IP: $EC2_IP"
echo "Key: $KEY_PATH"
echo "=========================================="

# Step 1: Create training directory on EC2
echo ""
echo "[1/4] Creating training directory on EC2..."
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "mkdir -p ~/training"

# Step 2: Transfer files
echo ""
echo "[2/4] Transferring files to EC2..."

echo "  - Transferring requirements folder..."
scp -i "$KEY_PATH" -r requirements "$EC2_USER@$EC2_IP:~/training/"

echo "  - Transferring training script..."
scp -i "$KEY_PATH" train_and_evaluate_on_ec2.py "$EC2_USER@$EC2_IP:~/training/"

echo "  ✓ All files transferred"

# Step 3: Run setup on EC2
echo ""
echo "[3/4] Running setup on EC2..."
echo "  This will take 5-10 minutes..."

ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" << 'EOF'
cd ~/training/requirements
chmod +x setup_ec2.sh
./setup_ec2.sh
EOF

echo "  ✓ Setup complete"

# Step 4: Verify installation
echo ""
echo "[4/4] Verifying installation..."
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" << 'EOF'
cd ~/training
source training_env/bin/activate
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
python -c "from b_box_utils import PreprocessorMarginRight; print('✅ b-box-utils OK')"
EOF

echo ""
echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo ""
echo "  ssh -i $KEY_PATH $EC2_USER@$EC2_IP"
echo "  cd ~/training && source training_env/bin/activate"
echo "  nohup python train_and_evaluate_on_ec2.py --s3-bucket YOUR_BUCKET_NAME 2>&1 | tee training.log &"
echo "  tail -f training.log"
echo ""
