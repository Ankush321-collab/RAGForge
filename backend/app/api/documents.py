from uuid import uuid4
from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status
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


@router.get("/download/{object_key:path}")
def download_document(object_key: str) -> Response:
	try:
		content = storage.download_bytes(object_key)
		media_type = "application/pdf" if object_key.endswith(".pdf") else "application/octet-stream"
		if object_key.endswith(".png"):
			media_type = "image/png"
		elif object_key.endswith(".jpg") or object_key.endswith(".jpeg"):
			media_type = "image/jpeg"
		return Response(content=content, media_type=media_type)
	except FileNotFoundError:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Object not found")


@router.get("", response_model=list[DocumentUploadResponse])
def list_documents() -> list[DocumentUploadResponse]:
	docs = []
	doc_dir = storage.local_dir / "documents"
	if doc_dir.exists():
		for d_id_dir in doc_dir.iterdir():
			if d_id_dir.is_dir():
				for f in d_id_dir.iterdir():
					if f.is_file():
						obj_key = f"documents/{d_id_dir.name}/{f.name}"
						docs.append(DocumentUploadResponse(
							document_id=d_id_dir.name,
							object_key=obj_key,
							filename=f.name,
							content_type="application/pdf" if f.name.endswith(".pdf") else "application/octet-stream",
							size_bytes=f.stat().st_size,
						))
	return docs
