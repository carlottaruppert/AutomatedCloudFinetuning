# Cloud Model Finetuning Pipeline

Automated pipeline for finetuning a regression-based model on AWS EC2 using transfer learning.

The project is intentionally template-like: you provide your own model artifact, dataset manifest, and S3 paths.

---

## Repository Structure

```text
AutomatedCloudFinetuning/
├── README.md
├── QUICK_START.md
├── s3_to_s3_filter.py               # Optional S3 data filtering/copy helper
├── train_and_evaluate_on_ec2.py     # Training + evaluation pipeline
├── deploy_to_ec2.sh                 # EC2 deployment automation
├── requirements/
│   ├── setup_ec2.sh
│   ├── requirements_ec2.txt         # TensorFlow + CUDA dependencies
│   └── b_box_utils-1.4.6-py3-none-any.whl
└── model/
    └── model.h5                     # Example local model artifact path
```

---

## Quick Start

Use `QUICK_START.md` as the canonical runbook for first execution.

The quick-start file contains:
- Optional S3 data preparation
- EC2 deployment commands
- Training launch command

This README focuses on architecture, configuration, and troubleshooting details.

---

## Training Pipeline

1. Downloads base model, manifest CSV, and images from S3
2. Splits data into train/test sets (stratified by class)
3. Evaluates baseline model performance
4. Finetunes model with transfer learning:
   - Freezes early layers
   - Trains remaining layers
   - Uses regression loss (`mse`) and `mae` metric
5. Evaluates finetuned model on held-out test set
6. Uploads artifacts back to S3:
   - `results/finetuned_model_*.h5`
   - `results/confusion_matrix_*.png`
   - `results/training_history_*.png`
   - `results/results_*.json`

---

## Regression + Class Mapping

The training script keeps a regression-style architecture:

- Model output: one continuous value in `[0, 1]`
- Label encoding (4 classes): targets mapped to:
  - Class 0 -> 0.125
  - Class 1 -> 0.375
  - Class 2 -> 0.625
  - Class 3 -> 0.875
- Evaluation converts continuous predictions back to class IDs via thresholds

If your task uses different class definitions or thresholds, update:
- `class_to_continuous(...)`
- `continuous_to_class(...)`

in `train_and_evaluate_on_ec2.py`.

---

## Input Data Contract

The training manifest CSV is expected to include:
- `filename` (image filename under `--s3-images-prefix`)
- `label` (integer class label, default expected range `0-3`)

Example:

```csv
filename,label
img_0001.png,0
img_0002.png,2
img_0003.png,1
```

---

## EC2 Configuration

### Suggested Baseline
- GPU instance with CUDA support (instance type depends on model size)
- Ubuntu 24.04 LTS
- 50GB+ storage

### IAM Permissions
- `s3:GetObject` on input model/data paths
- `s3:PutObject` on output artifact path
- `kms:Decrypt` if bucket encryption is enabled

---

## Main Runtime Parameters

`train_and_evaluate_on_ec2.py` supports:

- `--s3-bucket` (required)
- `--aws-profile` (optional)
- `--s3-model-key` (default: `model/model.h5`)
- `--s3-csv-key` (default: `data/training_data.csv`)
- `--s3-images-prefix` (default: `data/images/`)
- `--s3-output-prefix` (default: `results/`)
- `--num-layers-to-freeze` (default: `20`)

Core training constants in code:

```python
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.1
PATIENCE = 5
```

---

## Manual Setup (if deployment script fails)

```bash
# Transfer files
scp -i ~/.ssh/your-keypair.pem -r requirements ubuntu@EC2_IP:~/training/
scp -i ~/.ssh/your-keypair.pem train_and_evaluate_on_ec2.py ubuntu@EC2_IP:~/training/

# SSH and run setup
ssh -i ~/.ssh/your-keypair.pem ubuntu@EC2_IP
cd ~/training/requirements
chmod +x setup_ec2.sh
./setup_ec2.sh
```

---

## Monitoring

```bash
# GPU status
nvidia-smi

# Training log
tail -f ~/training/training.log

# Process check
ps aux | grep train_and_evaluate
```

---

## Troubleshooting

### GPU not detected

```bash
lspci | grep -i nvidia
pip uninstall -y tensorflow
pip install tensorflow[and-cuda]==2.16.2
```

### Dependency import errors

```bash
cd ~/training
source training_env/bin/activate
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "from b_box_utils import PreprocessorMarginRight; print('OK')"
```

### Out of memory

Lower `BATCH_SIZE` in `train_and_evaluate_on_ec2.py`:

```python
BATCH_SIZE = 4
```

---

## Validation Signals

Training is usually healthy if logs show:
- GPU is detected
- Manifest and images download successfully
- Loss is finite and trending down over epochs
- Confusion matrix includes predictions for multiple classes

---

Last updated: 2026-03-01
