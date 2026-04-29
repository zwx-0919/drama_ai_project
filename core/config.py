from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Drama AI System"
    api_prefix: str = "/api/v1"
    redis_url: str = "redis://localhost:6379/0"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "drama_scripts"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "drama-ai"
    minio_secure: bool = False
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    embedding_model: str = "text-embedding-v4"
    llm_model: str = "qwen-max"
    upload_dir: str = "docs/uploads"
    data_dir: str = "data"
    max_chunk_size: int = 1000
    chunk_overlap: int = 150

    @property
    def api_key(self) -> str:
        return self.dashscope_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
