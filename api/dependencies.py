from __future__ import annotations

from functools import lru_cache

from core.config import get_settings
from services.agent import ReActDramaAgent
from services.llm import ScriptEngine
from services.memory import RedisChatMessageHistory
from services.minio_storage import MinioStorage
from services.rag import RAGService


@lru_cache
def get_settings_dep():
    return get_settings()


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    return RAGService(
        settings.milvus_uri,
        settings.milvus_collection,
        settings.api_key,
        settings.redis_url,
    )


@lru_cache
def get_memory_service() -> RedisChatMessageHistory:
    settings = get_settings()
    return RedisChatMessageHistory(settings.redis_url)


@lru_cache
def get_script_engine() -> ScriptEngine:
    settings = get_settings()
    return ScriptEngine(api_key=settings.api_key, model_name=settings.llm_model)


@lru_cache
def get_minio_storage() -> MinioStorage:
    settings = get_settings()
    return MinioStorage(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket_name=settings.minio_bucket,
        secure=settings.minio_secure,
    )


@lru_cache
def get_agent() -> ReActDramaAgent:
    engine = get_script_engine()
    rag = get_rag_service()
    memory_service = get_memory_service()
    return ReActDramaAgent(engine, rag, memory_service)
