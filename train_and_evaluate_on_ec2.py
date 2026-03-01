#!/usr/bin/env python3
"""
S3-based model finetuning and evaluation on EC2.

This pipeline assumes a regression-style model that outputs a single continuous
value in [0, 1]. Predictions are mapped to 4 classes via configurable thresholds.

Training approach:
- Preprocess images with a production-compatible preprocessor
- Convert class labels (0-3) to continuous targets
- Finetune using MSE loss with early stopping
- Transform predictions back to classes for evaluation
"""

import os
import sys
import json
import boto3
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import time
from datetime import datetime
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Import production preprocessing
from b_box_utils import PreprocessorMarginRight

# Configuration
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.1
PATIENCE = 5
NUM_LAYERS_TO_FREEZE = 20

# Default S3 paths (override via CLI arguments)
DEFAULT_S3_MODEL_KEY = "model/model.h5"
DEFAULT_S3_CSV_KEY = "data/training_data.csv"
DEFAULT_S3_IMAGES_PREFIX = "data/images/"
DEFAULT_S3_OUTPUT_PREFIX = "results/"

# Local paths
LOCAL_DIR = os.path.expanduser("~/training")
LOCAL_MODEL_PATH = f"{LOCAL_DIR}/model.h5"
LOCAL_CSV_PATH = f"{LOCAL_DIR}/training_data.csv"
LOCAL_IMAGES_DIR = f"{LOCAL_DIR}/images"
LOCAL_OUTPUT_DIR = f"{LOCAL_DIR}/output"
FINETUNED_MODEL_PATH = f"{LOCAL_OUTPUT_DIR}/finetuned_model.h5"


def class_to_continuous(label: int) -> float:
    """
    Convert class label to continuous target value.
    
    Model outputs [0, 1], which is transformed to [-0.5, 3.5] in production.
    The transformation is: continuous_value = 4 * model_output - 0.5
    
    So for training, we need model targets in [0, 1] range:
    - Class 0 (A) → 0.125 (transforms to 0.0)
    - Class 1 (B) → 0.375 (transforms to 1.0)
    - Class 2 (C) → 0.625 (transforms to 2.0)
    - Class 3 (D) → 0.875 (transforms to 3.0)
    """
    return (label + 0.5) / 4.0


def continuous_to_class(continuous_value: float) -> int:
    """
    Convert continuous prediction to class using production thresholds.
    
    Production thresholds:
    - < 0.535 → Class 0 (A)
    - < 1.405 → Class 1 (B)
    - < 2.344 → Class 2 (C)
    - >= 2.344 → Class 3 (D)
    """
    # Transform from [0, 1] to [-0.5, 3.5] like in production
    transformed = 4 * continuous_value - 0.5
    
    if transformed < 0.535:
        return 0
    elif transformed < 1.405:
        return 1
    elif transformed < 2.344:
        return 2
    else:
        return 3


class S3ModelTrainer:
    def __init__(
        self,
        s3_bucket: str,
        aws_profile: str = None,
        s3_model_key: str = DEFAULT_S3_MODEL_KEY,
        s3_csv_key: str = DEFAULT_S3_CSV_KEY,
        s3_images_prefix: str = DEFAULT_S3_IMAGES_PREFIX,
        s3_output_prefix: str = DEFAULT_S3_OUTPUT_PREFIX,
        num_layers_to_freeze: int = NUM_LAYERS_TO_FREEZE
    ):
        self.s3_bucket = s3_bucket
        self.aws_profile = aws_profile
        self.s3_model_key = s3_model_key
        self.s3_csv_key = s3_csv_key
        self.s3_images_prefix = s3_images_prefix.rstrip("/") + "/"
        self.s3_output_prefix = s3_output_prefix.rstrip("/") + "/"
        self.num_layers_to_freeze = num_layers_to_freeze
        
        # Initialize S3 client
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            self.s3_client = session.client('s3')
        else:
            self.s3_client = boto3.client('s3')
        
        # Initialize production preprocessor
        self.preprocessor = PreprocessorMarginRight((IMG_SIZE, IMG_SIZE))
        
        # Create local directories
        os.makedirs(LOCAL_DIR, exist_ok=True)
        os.makedirs(LOCAL_IMAGES_DIR, exist_ok=True)
        os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
        
        # Results storage
        self.results = {
            'original_model_metrics': {},
            'finetuned_model_metrics': {},
            'training_history': {},
            'split_info': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def download_from_s3(self, s3_key: str, local_path: str):
        """Download a file from S3."""
        if os.path.exists(local_path):
            print(f"  ⏭️  Skipping download (file already exists): {local_path}")
            return
        
        print(f"Downloading s3://{self.s3_bucket}/{s3_key} to {local_path}")
        self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
        print(f"  ✓ Downloaded")
    
    def upload_to_s3(self, local_path: str, s3_key: str):
        """Upload a file to S3."""
        print(f"Uploading {local_path} to s3://{self.s3_bucket}/{s3_key}")
        self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
        print(f"  ✓ Uploaded")
    
    def download_images_from_csv(self, df: pd.DataFrame):
        """Download images listed in CSV from S3."""
        print(f"\nDownloading {len(df)} images from S3...")
        
        downloaded = 0
        failed = 0
        
        for idx, row in df.iterrows():
            filename = row['filename']
            s3_image_key = f"{self.s3_images_prefix}{filename}"
            local_image_path = os.path.join(LOCAL_IMAGES_DIR, filename)
            
            try:
                if not os.path.exists(local_image_path):
                    self.s3_client.download_file(self.s3_bucket, s3_image_key, local_image_path)
                downloaded += 1
                
                if downloaded % 100 == 0:
                    print(f"  Downloaded {downloaded}/{len(df)} images...")
            except Exception as e:
                print(f"  ❌ Failed to download {filename}: {e}")
                failed += 1
        
        print(f"✓ Downloaded {downloaded} images ({failed} failed)")
        return downloaded, failed
    
    def load_and_preprocess_batch(self, image_paths: list, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess a batch of images using production PreprocessorMarginRight.
        
        PreprocessorMarginRight handles:
        - Aspect ratio preservation
        - Margin padding on the right side
        - Resizing to target dimensions
        - Normalization (if built-in)
        """
        batch_images = []
        
        for img_path in image_paths:
            try:
                # Load image as PIL Image
                img = Image.open(img_path)
                
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Use production preprocessing
                # flip=False (will be handled per image if needed based on view_position)
                # prepare_input_data returns shape (1, H, W, C)
                processed = self.preprocessor.prepare_input_data(img, channels=1, flip=False)
                
                # Remove batch dimension: (1, H, W, C) -> (H, W, C)
                processed = processed[0]
                
                batch_images.append(processed)
            except Exception as e:
                print(f"⚠️  Error loading {img_path}: {e}")
                # Use zeros as fallback (shouldn't happen often)
                batch_images.append(np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32))
        
        return np.array(batch_images, dtype=np.float32), labels
    
    def create_dataset_generator(self, df: pd.DataFrame, shuffle: bool = True):
        """Create a data generator using production preprocessing."""
        image_paths = [os.path.join(LOCAL_IMAGES_DIR, fname) for fname in df['filename'].values]
        continuous_labels = np.array([class_to_continuous(label) for label in df['label'].values], dtype=np.float32)
        
        indices = np.arange(len(image_paths))
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, len(indices), BATCH_SIZE):
                batch_indices = indices[i:i+BATCH_SIZE]
                batch_paths = [image_paths[idx] for idx in batch_indices]
                batch_labels = continuous_labels[batch_indices]
                
                batch_images, batch_labels = self.load_and_preprocess_batch(batch_paths, batch_labels)
                
                yield batch_images, batch_labels
    
    def prepare_datasets(self, df: pd.DataFrame, split_ratio: float = 0.5) -> Tuple:
        """Prepare train and test datasets."""
        print(f"\n" + "=" * 80)
        print("DATA PREPARATION")
        print("=" * 80)
        
        # Remove rows where images don't exist
        valid_rows = []
        for idx, row in df.iterrows():
            image_path = os.path.join(LOCAL_IMAGES_DIR, row['filename'])
            if os.path.exists(image_path):
                valid_rows.append(idx)
        
        df = df.loc[valid_rows].reset_index(drop=True)
        print(f"Valid images found: {len(df)}")
        
        # Split data
        train_df, test_df = train_test_split(
            df, 
            test_size=split_ratio, 
            random_state=42,
            stratify=df['label']
        )
        
        print(f"\nTrain set: {len(train_df)} images")
        print(f"Test set:  {len(test_df)} images")
        
        # Print class distribution
        print(f"\nTrain distribution:")
        print(train_df['label'].value_counts().sort_index())
        print(f"\nTest distribution:")
        print(test_df['label'].value_counts().sort_index())
        
        # Store split info
        self.results['split_info'] = {
            'total_images': len(df),
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_distribution': train_df['label'].value_counts().to_dict(),
            'test_distribution': test_df['label'].value_counts().to_dict()
        }
        
        print(f"\n✓ Datasets prepared (using PreprocessorMarginRight)")
        
        return train_df, test_df
    
    def evaluate_model(self, model: keras.Model, test_df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Evaluate model and return metrics."""
        print(f"\n" + "=" * 80)
        print(f"EVALUATING {model_name.upper()}")
        print("=" * 80)
        
        # Prepare test data
        image_paths = [os.path.join(LOCAL_IMAGES_DIR, fname) for fname in test_df['filename'].values]
        y_test = test_df['label'].values
        
        # Predict in batches
        y_pred_continuous = []
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i+BATCH_SIZE]
            batch_labels = np.zeros(len(batch_paths))  # Dummy labels
            batch_images, _ = self.load_and_preprocess_batch(batch_paths, batch_labels)
            
            batch_pred = model.predict(batch_images, verbose=0).flatten()
            y_pred_continuous.extend(batch_pred)
            
            if (i // BATCH_SIZE) % 50 == 0:
                print(f"  Evaluated {i}/{len(image_paths)} images...")
        
        y_pred_continuous = np.array(y_pred_continuous)
        
        # Convert to classes
        y_pred = np.array([continuous_to_class(val) for val in y_pred_continuous])
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['A', 'B', 'C', 'D']))
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        
        # Show continuous prediction stats
        print(f"\nContinuous Predictions Stats:")
        print(f"  Min: {y_pred_continuous.min():.4f}")
        print(f"  Max: {y_pred_continuous.max():.4f}")
        print(f"  Mean: {y_pred_continuous.mean():.4f}")
        print(f"  Std: {y_pred_continuous.std():.4f}")
        
        # Save confusion matrix plot
        self.plot_confusion_matrix(conf_matrix, model_name)
        
        metrics = {
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'continuous_stats': {
                'min': float(y_pred_continuous.min()),
                'max': float(y_pred_continuous.max()),
                'mean': float(y_pred_continuous.mean()),
                'std': float(y_pred_continuous.std())
            }
        }
        
        return metrics
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, model_name: str):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['A', 'B', 'C', 'D'],
                   yticklabels=['A', 'B', 'C', 'D'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        output_path = os.path.join(LOCAL_OUTPUT_DIR, f'confusion_matrix_{model_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved confusion matrix: {output_path}")
    
    def finetune_model(self, model: keras.Model, train_df: pd.DataFrame) -> keras.Model:
        """Finetune the model using regression."""
        print(f"\n" + "=" * 80)
        print("FINETUNING MODEL (REGRESSION MODE WITH PRODUCTION PREPROCESSING)")
        print("=" * 80)
        
        # Freeze early layers
        print(f"Freezing first {self.num_layers_to_freeze} layers...")
        for i, layer in enumerate(model.layers[:self.num_layers_to_freeze]):
            layer.trainable = False
        
        # Print trainable status
        trainable_count = sum([1 for layer in model.layers if layer.trainable])
        print(f"Trainable layers: {trainable_count}/{len(model.layers)}")
        
        # Compile model for REGRESSION
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        # Split train into train/val
        train_size = len(train_df)
        val_size = int(train_size * VALIDATION_SPLIT)
        
        train_df_split = train_df.iloc[val_size:].reset_index(drop=True)
        val_df = train_df.iloc[:val_size].reset_index(drop=True)
        
        # Calculate steps
        steps_per_epoch = len(train_df_split) // BATCH_SIZE
        validation_steps = len(val_df) // BATCH_SIZE
        
        # Create generators
        train_generator = self.create_dataset_generator(train_df_split, shuffle=True)
        val_generator = self.create_dataset_generator(val_df, shuffle=False)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                FINETUNED_MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print(f"\nStarting training...")
        print(f"  Epochs: {EPOCHS}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Loss: MSE (Mean Squared Error)")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Validation steps: {validation_steps}")
        
        start_time = time.time()
        
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.1f}s")
        
        # Store training history
        self.results['training_history'] = {
            'loss': [float(x) for x in history.history['loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_mae': [float(x) for x in history.history['val_mae']],
            'epochs_trained': len(history.history['loss']),
            'training_time_seconds': training_time
        }
        
        # Plot training history
        self.plot_training_history(history)
        
        # Load best weights
        print(f"\nLoading best model from {FINETUNED_MODEL_PATH}")
        model = keras.models.load_model(FINETUNED_MODEL_PATH, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def plot_training_history(self, history):
        """Plot and save training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Train Loss (MSE)')
        ax1.plot(history.history['val_loss'], label='Val Loss (MSE)')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(history.history['mae'], label='Train MAE')
        ax2.plot(history.history['val_mae'], label='Val MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        output_path = os.path.join(LOCAL_OUTPUT_DIR, 'training_history.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved training history: {output_path}")
    
    def save_results(self):
        """Save results to JSON."""
        results_path = os.path.join(LOCAL_OUTPUT_DIR, 'results.json')
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Saved results: {results_path}")
    
    def upload_outputs_to_s3(self):
        """Upload all output files to S3."""
        print(f"\n" + "=" * 80)
        print("UPLOADING RESULTS TO S3")
        print("=" * 80)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        files_to_upload = [
            ('results.json', f'{self.s3_output_prefix}results_{timestamp}.json'),
            ('finetuned_model.h5', f'{self.s3_output_prefix}finetuned_model_{timestamp}.h5'),
            ('training_history.png', f'{self.s3_output_prefix}training_history_{timestamp}.png'),
            ('confusion_matrix_original_model.png', f'{self.s3_output_prefix}confusion_matrix_original_{timestamp}.png'),
            ('confusion_matrix_finetuned_model.png', f'{self.s3_output_prefix}confusion_matrix_finetuned_{timestamp}.png'),
        ]
        
        uploaded = []
        for local_filename, s3_key in files_to_upload:
            local_path = os.path.join(LOCAL_OUTPUT_DIR, local_filename)
            if os.path.exists(local_path):
                try:
                    self.upload_to_s3(local_path, s3_key)
                    uploaded.append(s3_key)
                except Exception as e:
                    print(f"  ❌ Failed to upload {local_filename}: {e}")
        
        print(f"\n✓ Uploaded {len(uploaded)} files to S3")
        return uploaded
    
    def run(self):
        """Main execution flow."""
        start_time = time.time()
        
        print("=" * 80)
        print("EC2 MODEL FINETUNING & EVALUATION")
        print("=" * 80)
        print(f"S3 Bucket: {self.s3_bucket}")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Preprocessor: PreprocessorMarginRight")
        print("=" * 80)
        
        # Step 1: Download model from S3
        print(f"\n[1/8] Downloading model from S3...")
        self.download_from_s3(self.s3_model_key, LOCAL_MODEL_PATH)
        
        # Step 2: Download CSV from S3
        print(f"\n[2/8] Downloading CSV from S3...")
        self.download_from_s3(self.s3_csv_key, LOCAL_CSV_PATH)
        
        # Step 3: Load CSV
        print(f"\n[3/8] Loading CSV...")
        df = pd.read_csv(LOCAL_CSV_PATH)
        print(f"  Loaded {len(df)} rows")
        
        # Step 4: Download images
        print(f"\n[4/8] Downloading images...")
        downloaded, failed = self.download_images_from_csv(df)
        
        # Step 5: Prepare datasets
        print(f"\n[5/8] Preparing datasets...")
        train_df, test_df = self.prepare_datasets(df, split_ratio=0.5)
        
        # Step 6: Load and evaluate original model
        print(f"\n[6/8] Evaluating original model...")
        original_model = keras.models.load_model(LOCAL_MODEL_PATH, compile=False)
        original_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        original_metrics = self.evaluate_model(original_model, test_df, "original_model")
        self.results['original_model_metrics'] = original_metrics
        
        # Step 7: Finetune model
        print(f"\n[7/8] Finetuning model...")
        finetuned_model = self.finetune_model(original_model, train_df)
        
        # Step 8: Evaluate finetuned model
        print(f"\n[8/8] Evaluating finetuned model...")
        finetuned_metrics = self.evaluate_model(finetuned_model, test_df, "finetuned_model")
        self.results['finetuned_model_metrics'] = finetuned_metrics
        
        # Save results
        self.save_results()
        
        # Upload to S3
        uploaded_files = self.upload_outputs_to_s3()
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total execution time: {total_time/60:.1f} minutes")
        print(f"\nOriginal Model Accuracy: {original_metrics['accuracy']:.4f}")
        print(f"Finetuned Model Accuracy: {finetuned_metrics['accuracy']:.4f}")
        print(f"Improvement: {(finetuned_metrics['accuracy'] - original_metrics['accuracy']):.4f}")
        print(f"\nUploaded {len(uploaded_files)} files to S3")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Finetune a regression-based model on S3-hosted data"
    )
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name")
    parser.add_argument("--aws-profile", default=None, help="AWS profile (optional)")
    parser.add_argument("--s3-model-key", default=DEFAULT_S3_MODEL_KEY, help=f"S3 key for base model (default: {DEFAULT_S3_MODEL_KEY})")
    parser.add_argument("--s3-csv-key", default=DEFAULT_S3_CSV_KEY, help=f"S3 key for training CSV (default: {DEFAULT_S3_CSV_KEY})")
    parser.add_argument("--s3-images-prefix", default=DEFAULT_S3_IMAGES_PREFIX, help=f"S3 prefix for image files (default: {DEFAULT_S3_IMAGES_PREFIX})")
    parser.add_argument("--s3-output-prefix", default=DEFAULT_S3_OUTPUT_PREFIX, help=f"S3 prefix for output artifacts (default: {DEFAULT_S3_OUTPUT_PREFIX})")
    parser.add_argument("--num-layers-to-freeze", type=int, default=NUM_LAYERS_TO_FREEZE, help=f"Number of initial layers to freeze (default: {NUM_LAYERS_TO_FREEZE})")
    
    args = parser.parse_args()
    
    # Check GPU availability
    print("\n" + "=" * 80)
    print("SYSTEM INFO")
    print("=" * 80)
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU available: {gpus}")
    if gpus:
        print(f"✅ GPU will be used for training!")
    else:
        print(f"⚠️  No GPU found, will use CPU (slower)")
    print("=" * 80)
    
    try:
        trainer = S3ModelTrainer(
            s3_bucket=args.s3_bucket,
            aws_profile=args.aws_profile,
            s3_model_key=args.s3_model_key,
            s3_csv_key=args.s3_csv_key,
            s3_images_prefix=args.s3_images_prefix,
            s3_output_prefix=args.s3_output_prefix,
            num_layers_to_freeze=args.num_layers_to_freeze,
        )
        trainer.run()
        print("\n✓ Training completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
