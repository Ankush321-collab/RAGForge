from pathlib import Path
from fastapi import APIRouter
from backend.app.core.config import get_settings
from backend.app.storage.s3 import S3Storage

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    settings = get_settings()
    storage = S3Storage()

    storage_status = "local" if storage.use_local else "s3"
    local_dir_exists = Path(settings.local_storage_dir).exists()

    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "environment": settings.environment,
        "storage_backend": storage_status,
        "local_storage_ready": local_dir_exists,
    }
