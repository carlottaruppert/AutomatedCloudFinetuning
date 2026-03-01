#!/usr/bin/env python3
"""
S3-to-S3 annotation filter and copier.

This script scans a source S3 bucket for PNG/JSON pairs, extracts labels from a
known annotation schema, and copies matched PNG files to a destination bucket
(entirely within S3, without local image downloads).

Current filtering logic:
- Looks at `evaluations -> <label_key>` (default: `density`)
- Accepts entries with a reannotation key (any key besides `bbox`)
- Accepts entries where `bbox -> confirmed_by` is not null
- Ignores entries that only have unconfirmed `bbox`

Usage:
    python s3_to_s3_filter.py SOURCE_BUCKET DEST_BUCKET --aws-profile PROFILE [OPTIONS]

Examples:
    # Copy all valid images
    python s3_to_s3_filter.py source-bucket destination-bucket --aws-profile my-profile

    # Copy specific IDs with a custom destination prefix
    python s3_to_s3_filter.py source-bucket destination-bucket --aws-profile my-profile --client-ids "123,456" --dest-prefix "data/images/"

    # Copy only latest exports
    python s3_to_s3_filter.py source-bucket destination-bucket --aws-profile my-profile --latest-export
"""

import os
import sys
import json
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError, NoCredentialsError
import argparse
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import csv


# Default mapping from class name to numeric label
CLASS_LABEL_MAP = {
    'A': 0,  # Almost entirely fatty
    'B': 1,  # Scattered fibroglandular densities
    'C': 2,  # Heterogeneously dense
    'D': 3   # Extremely dense
}

DEFAULT_LABEL_KEY = "density"
DEFAULT_LABEL_FIELD = "DensClass"


class S3ToS3LabelCopier:
    def __init__(self, source_bucket: str, dest_bucket: str, 
                 source_prefix: str = "", dest_prefix: str = "",
                 client_ids: List[str] = None, aws_profile: str = None, 
                 latest_export: bool = False, max_workers: int = 10,
                 csv_output_path: str = None,
                 label_key: str = DEFAULT_LABEL_KEY,
                 label_field: str = DEFAULT_LABEL_FIELD):
        if not aws_profile:
            raise ValueError("AWS profile must be specified")
        
        self.source_bucket = source_bucket
        self.dest_bucket = dest_bucket
        self.source_prefix = source_prefix.rstrip('/') + '/' if source_prefix else ""
        self.dest_prefix = dest_prefix.rstrip('/') + '/' if dest_prefix else ""
        self.client_ids = client_ids or []
        self.aws_profile = aws_profile
        self.latest_export = latest_export
        self.max_workers = max_workers
        self.csv_output_path = csv_output_path or "training_data.csv"
        self.label_key = label_key
        self.label_field = label_field
        
        # Initialize S3 client
        try:
            session = boto3.Session(profile_name=self.aws_profile)
            self.s3_client = session.client('s3')
            
            # Verify source bucket access
            self.s3_client.head_bucket(Bucket=source_bucket)
            
            # Verify destination bucket access
            self.s3_client.head_bucket(Bucket=dest_bucket)
            
        except NoCredentialsError:
            raise Exception("AWS credentials not found. Please configure your credentials.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '403':
                raise Exception(f"Access denied to bucket. Check your permissions.")
            elif error_code == '404':
                raise Exception(f"Bucket not found.")
            else:
                raise Exception(f"Error accessing bucket: {e}")
        
        # Statistics
        self.stats = {
            'total_json_files': 0,
            'valid_annotations': 0,
            'images_copied': 0,
            'images_failed': 0,
            'class_counts': {class_name: 0 for class_name in CLASS_LABEL_MAP}
        }
        
        # For CSV generation
        self.processed_images = []
        self.csv_s3_path = None
    
    def get_latest_upload_dir(self, client_id: str) -> Optional[str]:
        """Find the latest uploaded_at directory for a client."""
        try:
            prefix = f"{self.source_prefix}{client_id}/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.source_bucket, Prefix=prefix, Delimiter='/')
            
            upload_dirs = []
            for page in pages:
                if 'CommonPrefixes' in page:
                    for common_prefix in page['CommonPrefixes']:
                        dir_name = common_prefix['Prefix']
                        if 'uploaded_at_' in dir_name:
                            upload_dirs.append(dir_name)
            
            if upload_dirs:
                latest = sorted(upload_dirs, reverse=True)[0]
                return latest
            return None
        except ClientError as e:
            print(f"Error finding latest upload for client {client_id}: {e}")
            return None
    
    def list_json_files(self) -> List[str]:
        """List all JSON files in source bucket based on configuration."""
        json_files = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            if self.client_ids:
                print(f"Scanning {len(self.client_ids)} client(s): {', '.join(self.client_ids)}")
                
                for client_id in self.client_ids:
                    if self.latest_export:
                        latest_dir = self.get_latest_upload_dir(client_id)
                        if latest_dir:
                            print(f"  Latest export for {client_id}: {latest_dir}")
                            search_prefix = latest_dir
                        else:
                            print(f"  No uploads found for {client_id}")
                            continue
                    else:
                        search_prefix = f"{self.source_prefix}{client_id}/"
                    
                    print(f"  Scanning: {search_prefix}")
                    
                    for page in paginator.paginate(Bucket=self.source_bucket, Prefix=search_prefix):
                        if 'Contents' in page:
                            for obj in page['Contents']:
                                key = obj['Key']
                                if key.endswith('.json'):
                                    json_files.append(key)
            else:
                print(f"Scanning entire bucket (all clients/uploads)")
                paginate_kwargs = {'Bucket': self.source_bucket}
                if self.source_prefix:
                    paginate_kwargs['Prefix'] = self.source_prefix
                
                for page in paginator.paginate(**paginate_kwargs):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            if key.endswith('.json'):
                                json_files.append(key)
        
        except ClientError as e:
            raise Exception(f"Error listing objects in bucket: {e}")
        
        print(f"Found {len(json_files)} JSON files")
        return json_files
    
    def extract_label(self, json_data: Dict) -> Optional[Tuple[str, str]]:
        """
        Extract label from JSON if it's confirmed or reannotated.
        
        Returns:
            Tuple of (class_name, annotation_source) or None if not valid
        """
        try:
            evaluations = json_data.get('evaluations', {})
            label_data = evaluations.get(self.label_key, {})
            
            if not label_data:
                return None
            
            # Check for reannotation keys (any key besides 'bbox')
            reannotation_keys = [k for k in label_data.keys() if k != 'bbox']
            
            if reannotation_keys:
                reannotation_key = reannotation_keys[0]
                reannotation = label_data[reannotation_key]
                class_name = reannotation.get(self.label_field)
                if class_name in CLASS_LABEL_MAP:
                    return (class_name, f'reannotated_{reannotation_key}')
            
            # Check bbox for confirmed_by
            bbox = label_data.get('bbox', {})
            if bbox.get('confirmed_by') is not None:
                class_name = bbox.get(self.label_field)
                if class_name in CLASS_LABEL_MAP:
                    return (class_name, f"confirmed_by_{bbox['confirmed_by']}")
            
            return None
            
        except (KeyError, TypeError) as e:
            print(f"Error parsing label data: {e}")
            return None
    
    def get_png_path_from_json_path(self, json_path: str) -> str:
        """Convert JSON path to corresponding PNG path."""
        return json_path.replace('.json', '.png')
    
    def upload_csv_to_s3(self):
        """Upload the CSV file to S3 destination bucket."""
        try:
            csv_filename = Path(self.csv_output_path).name
            s3_csv_key = f"{self.dest_prefix}{csv_filename}"
            
            print(f"\n📤 Uploading CSV to S3...")
            self.s3_client.upload_file(
                self.csv_output_path,
                self.dest_bucket,
                s3_csv_key
            )
            print(f"   ✓ Uploaded to: s3://{self.dest_bucket}/{s3_csv_key}")
            
            # Store for summary
            self.csv_s3_path = f"s3://{self.dest_bucket}/{s3_csv_key}"
            
        except Exception as e:
            print(f"   ❌ Error uploading CSV to S3: {e}")
            self.csv_s3_path = None
    
    def copy_png_s3_to_s3(self, source_key: str, dest_key: str) -> bool:
        """
        Copy PNG from source bucket to destination bucket within S3.
        """
        try:
            copy_source = {
                'Bucket': self.source_bucket,
                'Key': source_key
            }
            
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.dest_bucket,
                Key=dest_key
            )
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"  ❌ Source file not found: {source_key}")
            else:
                print(f"  ❌ Error copying {source_key}: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Unexpected error copying {source_key}: {e}")
            return False
    
    def process_image(self, json_key: str, class_name: str, 
                     annotation_source: str) -> Optional[Dict[str, Any]]:
        """
        Copy PNG from source to destination S3 bucket.
        """
        try:
            # Get PNG path in source bucket
            source_png_key = self.get_png_path_from_json_path(json_key)
            
            # Create destination path
            # Use just the filename or preserve structure based on dest_prefix
            png_filename = Path(source_png_key).name
            dest_png_key = f"{self.dest_prefix}{png_filename}"
            
            # Check if destination already exists (optional optimization)
            try:
                self.s3_client.head_object(Bucket=self.dest_bucket, Key=dest_png_key)
                print(f"  ⏭️  Skipping {png_filename} (already exists in destination)")
                return {
                    'filename': png_filename,
                    'label_value': class_name,
                    'label': CLASS_LABEL_MAP[class_name],
                    'source': annotation_source,
                    'source_s3_path': source_png_key,
                    'dest_s3_path': dest_png_key
                }
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    # Some other error, not "not found"
                    raise
                # File doesn't exist in destination, proceed with copy
            
            # Copy PNG from source to destination
            if not self.copy_png_s3_to_s3(source_png_key, dest_png_key):
                return None
            
            print(f"  ✓ Copied: {png_filename} (label: {class_name}={CLASS_LABEL_MAP[class_name]})")
            
            return {
                'filename': png_filename,
                'label_value': class_name,
                'label': CLASS_LABEL_MAP[class_name],
                'source': annotation_source,
                'source_s3_path': source_png_key,
                'dest_s3_path': dest_png_key
            }
            
        except Exception as e:
            print(f"  ❌ Error processing {json_key}: {e}")
            return None
    
    def process_json_file(self, json_key: str) -> Optional[Dict[str, Any]]:
        """Download and process a single JSON file."""
        try:
            # Download JSON
            response = self.s3_client.get_object(Bucket=self.source_bucket, Key=json_key)
            json_content = response['Body'].read().decode('utf-8')
            json_data = json.loads(json_content)
            
            # Extract label
            result = self.extract_label(json_data)
            if result is None:
                return None
            
            class_name, annotation_source = result
            
            print(f"✓ Found valid annotation: {json_key}")
            print(f"  Label: {class_name} ({CLASS_LABEL_MAP[class_name]}), Source: {annotation_source}")
            
            # Update stats
            self.stats['valid_annotations'] += 1
            self.stats['class_counts'][class_name] += 1
            
            # Copy image
            image_metadata = self.process_image(json_key, class_name, annotation_source)
            
            if image_metadata:
                self.stats['images_copied'] += 1
                return image_metadata
            else:
                self.stats['images_failed'] += 1
                return None
                
        except Exception as e:
            print(f"Error processing {json_key}: {e}")
            return None
    
    def save_csv(self, image_metadata_list: List[Dict[str, Any]]):
        """Save training metadata to CSV file (locally and upload to S3)."""
        if not image_metadata_list:
            print("No images to save to CSV")
            return
        
        df = pd.DataFrame(image_metadata_list)
        df = df[['filename', 'label', 'label_value', 'source', 'dest_s3_path']]
        df.to_csv(self.csv_output_path, index=False)
        
        print(f"\n📄 Saved CSV locally: {self.csv_output_path}")
        print(f"   Columns: {', '.join(df.columns)}")
        print(f"   Total rows: {len(df)}")
        
        # Upload CSV to S3
        self.upload_csv_to_s3()
    
    def run(self):
        """Main execution flow."""
        start_time = time.time()
        
        print("=" * 80)
        print("S3-to-S3 Label Data Filter & Copier")
        print("=" * 80)
        print(f"Source bucket: {self.source_bucket}")
        print(f"Destination bucket: {self.dest_bucket}")
        print(f"Destination prefix: {self.dest_prefix or '(root)'}")
        print(f"CSV output: {self.csv_output_path}")
        print(f"Label key: evaluations -> {self.label_key}")
        print(f"Label field: {self.label_field}")
        print("=" * 80)
        
        # List JSON files
        print(f"\nScanning bucket '{self.source_bucket}' for JSON files...")
        json_files = self.list_json_files()
        self.stats['total_json_files'] = len(json_files)
        
        if not json_files:
            print("No JSON files found!")
            return
        
        print(f"\nProcessing {len(json_files)} JSON files...")
        print("=" * 80)
        
        # Process files with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_json_file, json_key): json_key 
                      for json_key in json_files}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.processed_images.append(result)
        
        # Save CSV
        if self.processed_images:
            self.save_csv(self.processed_images)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total JSON files scanned: {self.stats['total_json_files']}")
        print(f"Valid annotations found: {self.stats['valid_annotations']}")
        print(f"Images successfully copied: {self.stats['images_copied']}")
        print(f"Images failed: {self.stats['images_failed']}")
        print(f"Time elapsed: {elapsed_time:.1f}s")
        
        print("\nClass distribution:")
        for class_name in CLASS_LABEL_MAP:
            count = self.stats['class_counts'][class_name]
            label = CLASS_LABEL_MAP[class_name]
            print(f"  {class_name} (label {label}): {count}")
        
        print(f"\nDestination:")
        print(f"  S3 Bucket: s3://{self.dest_bucket}/{self.dest_prefix}")
        print(f"  Local CSV: {self.csv_output_path}")
        if self.csv_s3_path:
            print(f"  S3 CSV: {self.csv_s3_path}")
        print("=" * 80)


def parse_client_ids(client_ids_str: str) -> List[str]:
    """Parse client IDs from format like '(123, 456)' or comma-separated values."""
    if not client_ids_str:
        return []
    
    client_ids_str = client_ids_str.strip()
    if client_ids_str.startswith('(') and client_ids_str.endswith(')'):
        client_ids_str = client_ids_str[1:-1]
    
    client_ids = [client_id.strip() for client_id in client_ids_str.split(',')]
    client_ids = [client_id for client_id in client_ids if client_id]
    
    return client_ids


def main():
    parser = argparse.ArgumentParser(
        description="Filter label annotations and copy matching images between S3 buckets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy all filtered images to destination bucket
  %(prog)s source-bucket dest-bucket --aws-profile my-profile
  
  # Copy specific client IDs with custom destination prefix
  %(prog)s source-bucket dest-bucket --aws-profile my-profile --client-ids "123,456" --dest-prefix "data/images/"
  
  # Copy only latest exports
  %(prog)s source-bucket dest-bucket --aws-profile my-profile --latest-export
        """
    )
    
    parser.add_argument("source_bucket", help="Source S3 bucket name")
    parser.add_argument("dest_bucket", help="Destination S3 bucket name")
    parser.add_argument("--aws-profile", required=True, help="AWS profile to use")
    parser.add_argument("--source-prefix", default="", help="S3 prefix in source bucket to limit search scope")
    parser.add_argument("--dest-prefix", default="", help="S3 prefix in destination bucket (e.g., 'data/images/')")
    parser.add_argument("--client-ids", default="", 
                       help="Comma-separated list of client IDs to scan")
    parser.add_argument("--latest-export", action='store_true', default=False,
                       help="Only scan the latest upload directory for each client")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Maximum number of parallel operations (default: 10)")
    parser.add_argument("--csv-output", default="training_data.csv",
                       help="Path for output CSV file (default: training_data.csv)")
    parser.add_argument("--label-key", default=DEFAULT_LABEL_KEY,
                       help=f"Annotation key under evaluations (default: {DEFAULT_LABEL_KEY})")
    parser.add_argument("--label-field", default=DEFAULT_LABEL_FIELD,
                       help=f"Label field name inside annotation payloads (default: {DEFAULT_LABEL_FIELD})")
    
    args = parser.parse_args()
    
    # Parse client IDs
    client_ids = parse_client_ids(args.client_ids)
    
    print(f"🔑 Using AWS Profile: {args.aws_profile}")
    
    try:
        copier = S3ToS3LabelCopier(
            args.source_bucket,
            args.dest_bucket,
            args.source_prefix,
            args.dest_prefix,
            client_ids, 
            args.aws_profile, 
            args.latest_export,
            args.max_workers,
            args.csv_output,
            args.label_key,
            args.label_field
        )
        copier.run()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
