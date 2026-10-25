import time
import sys
import boto3

print("[MinIO-Init] Connecting to MinIO service...")
endpoint = "http://minio:9000"
bucket_name = "ragforge"

for attempt in range(1, 31):
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadminpassword",
            region_name="us-east-1",
        )
        existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
        if bucket_name not in existing:
            s3.create_bucket(Bucket=bucket_name)
            print(f"[MinIO-Init] Created bucket '{bucket_name}' successfully.")
        else:
            print(f"[MinIO-Init] Bucket '{bucket_name}' already exists.")
        sys.exit(0)
    except Exception as e:
        print(f"[MinIO-Init] Waiting for MinIO... (attempt {attempt}/30: {e})")
        time.sleep(2)

print("[MinIO-Init] ERROR: Failed to connect to MinIO after 30 attempts.")
sys.exit(1)
