# Quick Start Guide

## 1) Prepare Data (Optional)

Use this only if you need to filter/copy data into a training bucket first:

```bash
python s3_to_s3_filter.py SOURCE_BUCKET DEST_BUCKET \
  --aws-profile YOUR_AWS_PROFILE \
  --source-prefix raw/ \
  --dest-prefix data/images/
```

This creates a manifest CSV (default: `training_data.csv`) and uploads it to the destination bucket.

---

## 2) Deploy to EC2

```bash
chmod +x deploy_to_ec2.sh
./deploy_to_ec2.sh YOUR_EC2_IP ~/.ssh/your-keypair.pem
```

Time: ~5-10 minutes

---

## 3) Start Training

```bash
ssh -i ~/.ssh/your-keypair.pem ubuntu@YOUR_EC2_IP
cd ~/training
source training_env/bin/activate

nohup python train_and_evaluate_on_ec2.py \
  --s3-bucket YOUR_BUCKET_NAME \
  --s3-model-key model/model.h5 \
  --s3-csv-key data/training_data.csv \
  --s3-images-prefix data/images/ \
  --s3-output-prefix results/ \
  2>&1 | tee training.log &

tail -f training.log
```

Time: depends on dataset size, model size, and instance type.

---

See `README.md` for monitoring, infrastructure guidance, and troubleshooting.
