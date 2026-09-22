import re
from pathlib import PurePath

from fastapi import HTTPException, UploadFile, status

from backend.app.core.config import get_settings


def safe_filename(filename: str) -> str:
	name = PurePath(filename).name
	name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
	if not name:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")
	return name


def validate_upload(upload: UploadFile) -> str:
	settings = get_settings()
	filename = safe_filename(upload.filename or "")
	if PurePath(filename).suffix.lower() not in settings.allowed_upload_extensions:
		raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF uploads are supported")
	if upload.content_type not in {"application/pdf", "application/octet-stream"}:
		raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF uploads are supported")
	return filename


def validate_upload_size(size: int) -> None:
	maximum = get_settings().max_upload_size_bytes
	if size > maximum:
		raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File exceeds the {maximum} byte upload limit")
