from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from backend.app.core.config import get_settings
from backend.app.core.security import validate_upload, validate_upload_size
from backend.app.storage.s3 import S3Storage


router = APIRouter(prefix="/documents", tags=["documents"])
storage = S3Storage()


class DocumentUploadResponse(BaseModel):
	document_id: str
	object_key: str
	filename: str
	content_type: str
	size_bytes: int


class PresignedUrlResponse(BaseModel):
	document_id: str
	object_key: str
	url: str
	expires_in: int


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
	filename = validate_upload(file)
	content = await file.read()
	validate_upload_size(len(content))
	document_id = str(uuid4())
	object_key = storage.build_object_key(document_id, filename)
	storage.upload_bytes(content, object_key, file.content_type or "application/pdf")
	return DocumentUploadResponse(
		document_id=document_id,
		object_key=object_key,
		filename=filename,
		content_type=file.content_type or "application/pdf",
		size_bytes=len(content),
	)


@router.get("/{document_id}/url", response_model=PresignedUrlResponse)
def get_document_url(document_id: str) -> PresignedUrlResponse:
	object_key = storage.find_document_key(document_id)
	if object_key is None:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
	expiry = get_settings().s3_presigned_url_expiry
	return PresignedUrlResponse(
		document_id=document_id,
		object_key=object_key,
		url=storage.generate_presigned_url(object_key, expiry),
		expires_in=expiry,
	)
