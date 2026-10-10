from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from backend.app.storage.s3 import S3Storage


def test_s3_storage_generate_presigned_url():
    with patch("backend.app.storage.s3.boto3.client") as mock_boto_client:
        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        mock_client_instance.generate_presigned_url.return_value = "https://s3.amazonaws.com/ragforge-documents/documents/doc123/file.pdf?AWSAccessKeyId=MOCK&Signature=MOCK"

        storage = S3Storage()
        url = storage.generate_presigned_url("documents/doc123/file.pdf", expires_in=900)

        assert "https://s3.amazonaws.com" in url
        mock_client_instance.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": storage.bucket_name, "Key": "documents/doc123/file.pdf"},
            ExpiresIn=900,
        )


def test_get_document_presigned_url_success():
    from backend.app.api.documents import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    with patch("backend.app.api.documents.storage.find_document_key") as mock_find, \
         patch("backend.app.api.documents.storage.generate_presigned_url") as mock_gen:
        
        mock_find.return_value = "documents/test-id-123/sample.pdf"
        mock_gen.return_value = "http://localhost:9000/ragforge-documents/documents/test-id-123/sample.pdf?sig=123"

        response = client.get("/documents/test-id-123/url")

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "test-id-123"
        assert data["object_key"] == "documents/test-id-123/sample.pdf"
        assert data["url"] == "http://localhost:9000/ragforge-documents/documents/test-id-123/sample.pdf?sig=123"
        assert data["expires_in"] == 900

        mock_find.assert_called_once_with("test-id-123")
        mock_gen.assert_called_once_with("documents/test-id-123/sample.pdf", 900)


def test_get_document_presigned_url_not_found():
    from backend.app.api.documents import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    with patch("backend.app.api.documents.storage.find_document_key") as mock_find:
        mock_find.return_value = None

        response = client.get("/documents/non-existent-id/url")

        assert response.status_code == 404
        assert response.json()["detail"] == "Document not found"
