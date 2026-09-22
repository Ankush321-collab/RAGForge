from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	app_name: str = "Multimodal Agentic RAG"
	environment: str = "development"
	s3_endpoint_url: str | None = Field(
		default=None,
		validation_alias=AliasChoices("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL"),
	)
	s3_region: str = Field(
		default="us-east-1",
		validation_alias=AliasChoices("S3_REGION", "AWS_DEFAULT_REGION"),
	)
	s3_access_key_id: str = Field(
		validation_alias=AliasChoices("S3_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
	)
	s3_secret_access_key: str = Field(
		validation_alias=AliasChoices("S3_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
	)
	s3_bucket_name: str = Field(
		default="ragforge-documents",
		validation_alias=AliasChoices("S3_BUCKET_NAME", "AWS_S3_BUCKET"),
	)
	s3_presigned_url_expiry: int = 900
	max_upload_size_bytes: int = 50 * 1024 * 1024
	allowed_upload_extensions: tuple[str, ...] = (".pdf",)

	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		case_sensitive=False,
		extra="ignore",
	)


@lru_cache
def get_settings() -> Settings:
	return Settings()
