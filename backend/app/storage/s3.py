from io import BytesIO
import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


class S3Storage:
    """Manages document object storage supporting both AWS S3/MinIO and local filesystem fallback."""

    def __init__(self, backend: Optional[str] = None) -> None:
        settings = get_settings()
        self.bucket_name = settings.s3_bucket_name
        self.region = settings.s3_region
        self.local_dir = Path(settings.local_storage_dir) / self.bucket_name
        self.base_url = settings.base_url.rstrip("/")

        self.backend = backend or settings.storage_backend
        self.use_local = self.backend == "local"

        try:
            from botocore.config import Config
            client_config = Config(
                connect_timeout=2,
                read_timeout=2,
                retries={"max_attempts": 1},
                s3={"addressing_style": "path"},
            )
            self.client = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint_url,
                region_name=settings.s3_region,
                aws_access_key_id=settings.s3_access_key_id,
                aws_secret_access_key=settings.s3_secret_access_key,
                config=client_config,
            )
        except Exception as e:
            logger.warning("Failed to initialize boto3 client: %s. Using local filesystem storage.", e)
            self.client = None
            self.use_local = True

    @staticmethod
    def build_object_key(document_id: str, filename: str) -> str:
        return f"documents/{document_id}/{filename}"

    def _local_path(self, object_key: str) -> Path:
        clean_key = object_key.replace("\\", "/").lstrip("/")
        return self.local_dir / clean_key

    def ensure_bucket(self) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        if self.use_local or not self.client:
            return

        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except (ClientError, EndpointConnectionError, BotoCoreError):
            try:
                if self.region and self.region != "us-east-1":
                    self.client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": self.region},
                    )
                else:
                    self.client.create_bucket(Bucket=self.bucket_name)
            except ClientError as err:
                error_code = err.response.get("Error", {}).get("Code", "")
                if error_code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                    return
                logger.warning("S3 head/create bucket failed (%s); switching to local storage.", err)
                self.use_local = True
            except Exception as err:
                logger.warning("S3 head/create bucket failed (%s); switching to local storage.", err)
                self.use_local = True

    def upload_bytes(self, content: bytes, object_key: str, content_type: str = "application/pdf") -> None:
        if self.use_local or not self.client:
            path = self._local_path(object_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return

        try:
            self.client.upload_fileobj(
                BytesIO(content),
                self.bucket_name,
                object_key,
                ExtraArgs={"ContentType": content_type},
            )
        except Exception as e:
            logger.warning("S3 upload failed (%s). Falling back to local storage.", e)
            self.use_local = True
            path = self._local_path(object_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def download_bytes(self, object_key: str) -> bytes:
        if not self.use_local and self.client:
            try:
                response = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
                return response["Body"].read()
            except Exception as e:
                logger.warning("S3 download failed (%s). Attempting local retrieval.", e)

        path = self._local_path(object_key)
        if path.exists():
            return path.read_bytes()
        raise FileNotFoundError(f"Object {object_key} not found in S3 or local storage.")

    def generate_presigned_url(self, object_key: str, expires_in: int = 900) -> str:
        if self.client:
            try:
                return self.client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket_name, "Key": object_key},
                    ExpiresIn=expires_in,
                )
            except Exception as e:
                logger.warning("Presigned URL generation failed (%s); using local URL.", e)

        clean_key = object_key.replace("\\", "/").lstrip("/")
        return f"{self.base_url}/documents/download/{clean_key}"

    def find_document_key(self, document_id: str) -> Optional[str]:
        if not self.use_local and self.client:
            try:
                response = self.client.list_objects_v2(
                    Bucket=self.bucket_name, Prefix=f"documents/{document_id}/", MaxKeys=1
                )
                objects = response.get("Contents", [])
                if objects:
                    return objects[0]["Key"]
            except Exception as e:
                logger.warning("S3 list objects failed (%s). Falling back to local search.", e)

        doc_dir = self.local_dir / "documents" / document_id
        if doc_dir.exists():
            for item in doc_dir.glob("*"):
                if item.is_file():
                    return f"documents/{document_id}/{item.name}"
        return None

    def delete(self, object_key: str) -> None:
        if not self.use_local and self.client:
            try:
                self.client.delete_object(Bucket=self.bucket_name, Key=object_key)
            except Exception:
                pass
        path = self._local_path(object_key)
        if path.exists():
            path.unlink()