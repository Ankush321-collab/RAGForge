from io import BytesIO

import boto3
from botocore.exceptions import ClientError

from backend.app.core.config import get_settings


class S3Storage:
    def __init__(self) -> None:
        settings = get_settings()
        self.bucket_name = settings.s3_bucket_name
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
        )

    @staticmethod
    def build_object_key(document_id: str, filename: str) -> str:
        return f"documents/{document_id}/{filename}"

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket_name) 
            
    def upload_bytes(self, content: bytes, object_key: str, content_type: str) -> None:
        self.client.upload_fileobj(
            BytesIO(content), self.bucket_name, object_key,
            ExtraArgs={"ContentType": content_type},
        )

    def download_bytes(self, object_key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
        return response["Body"].read()

    def generate_presigned_url(self, object_key: str, expires_in: int) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": object_key},
            ExpiresIn=expires_in,
        )

    def find_document_key(self, document_id: str) -> str | None:
        response = self.client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=f"documents/{document_id}/", MaxKeys=1
        )
        objects = response.get("Contents", [])
        return objects[0]["Key"] if objects else None

    def delete(self, object_key: str) -> None:
        self.client.delete_object(Bucket=self.bucket_name, Key=object_key)