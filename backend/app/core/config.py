from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	app_name: str = "Multimodal Agentic RAG"
	environment: str = "development"
	storage_backend: str = Field(
		default="auto",
		validation_alias=AliasChoices("STORAGE_BACKEND", "STORAGE_TYPE"),
	)
	local_storage_dir: str = Field(
		default="data/storage",
		validation_alias=AliasChoices("LOCAL_STORAGE_DIR", "STORAGE_LOCAL_DIR"),
	)
	base_url: str = Field(
		default="http://localhost:8000",
		validation_alias=AliasChoices("BASE_URL", "APP_BASE_URL"),
	)
	s3_endpoint_url: str | None = Field(
		default=None,
		validation_alias=AliasChoices("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL"),
	)
	s3_region: str = Field(
		default="ap-south-2",
		validation_alias=AliasChoices("S3_REGION", "AWS_DEFAULT_REGION"),
	)
	s3_access_key_id: str = Field(
		default="mock_access_key",
		validation_alias=AliasChoices("S3_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
	)
	s3_secret_access_key: str = Field(
		default="mock_secret_key",
		validation_alias=AliasChoices("S3_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
	)
	s3_bucket_name: str = Field(
		default="ragforge-documents",
		validation_alias=AliasChoices("S3_BUCKET_NAME", "AWS_S3_BUCKET"),
	)
	s3_presigned_url_expiry: int = 900
	max_upload_size_bytes: int = 50 * 1024 * 1024
	allowed_upload_extensions: tuple[str, ...] = (".pdf",)

	opensearch_host: str | None = Field(
		default=None,
		validation_alias=AliasChoices("OPENSEARCH_HOST", "AWS_OPENSEARCH_ENDPOINT"),
	)
	opensearch_username: str | None = Field(
		default="admin",
		validation_alias=AliasChoices("OPENSEARCH_USERNAME", "AWS_OPENSEARCH_USER"),
	)
	opensearch_password: str | None = Field(
		default=None,
		validation_alias=AliasChoices("OPENSEARCH_PASSWORD", "AWS_OPENSEARCH_PASSWORD"),
	)
	opensearch_index_name: str = Field(
		default="ragforge-chunks",
		validation_alias=AliasChoices("OPENSEARCH_INDEX_NAME", "AWS_OPENSEARCH_INDEX"),
	)

	neo4j_uri: str | None = Field(
		default=None,
		validation_alias=AliasChoices("NEO4J_URI", "NEO4J_URL"),
	)
	neo4j_username: str | None = Field(
		default="neo4j",
		validation_alias=AliasChoices("NEO4J_USERNAME", "NEO4J_USER"),
	)
	neo4j_password: str | None = Field(
		default=None,
		validation_alias=AliasChoices("NEO4J_PASSWORD"),
	)

	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		case_sensitive=False,
		extra="ignore",
	)


@lru_cache
def get_settings() -> Settings:
	return Settings()
