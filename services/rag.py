from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

try:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
except Exception:  # pragma: no cover
    Collection = None
    CollectionSchema = None
    DataType = None
    FieldSchema = None
    connections = None
    utility = None

try:
    from redis import Redis
except Exception:  # pragma: no cover
    Redis = None

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    Document = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None

from utils.text import split_text


@dataclass
class SearchResult:
    script_id: str
    content: str
    score: float
    user_id: str
    source: str = "milvus"


class RAGService:
    """Redis-backed cache + Milvus vector store service."""

    def __init__(
        self,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "drama_scripts",
        api_key: str = "",
        redis_url: str = "redis://localhost:6379/0",
        embedding_dim: int = 128,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._redis = Redis.from_url(redis_url, decode_responses=True) if Redis is not None and redis_url else None
        self._milvus_ready = Collection is not None and connections is not None and utility is not None
        self._collection = None
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) if RecursiveCharacterTextSplitter is not None else None
        self._connect_milvus(milvus_uri)
        self._legacy_memory: dict[str, dict[str, str]] = {}

    def _connect_milvus(self, milvus_uri: str) -> None:
        if not self._milvus_ready:
            return
        host, port = self._parse_milvus_uri(milvus_uri)
        try:
            connections.connect(alias="default", host=host, port=port)
        except Exception:
            self._collection = None
            return
        try:
            if utility.has_collection(self.collection_name):
                self._collection = Collection(self.collection_name)
                self._collection.load()
                return
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="script_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields, description="Drama scripts vector store")
            self._collection = Collection(self.collection_name, schema=schema)
            self._collection.create_index(field_name="embedding", index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}})
            self._collection.load()
        except Exception:
            self._collection = None

    @staticmethod
    def _parse_milvus_uri(milvus_uri: str) -> tuple[str, int]:
        value = milvus_uri.replace("http://", "").replace("https://", "")
        if ":" in value:
            host, port_text = value.split(":", 1)
            return host, int(port_text)
        return value, 19530

    def _redis_key(self, *parts: str) -> str:
        return ":".join(("drama", *parts))

    def _cache_get_json(self, key: str) -> Optional[Any]:
        if not self._redis:
            return None
        payload = self._redis.get(key)
        return json.loads(payload) if payload else None

    def _cache_set_json(self, key: str, value: Any, ttl: int = 3600) -> None:
        if self._redis:
            self._redis.setex(key, ttl, json.dumps(value, ensure_ascii=False))

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower())
        vec = [0.0] * self.embedding_dim
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, min(len(digest), self.embedding_dim)):
                vec[i] += digest[i] / 255.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _serialize_result(self, result: SearchResult) -> dict[str, Any]:
        return {"script_id": result.script_id, "content": result.content, "score": result.score, "user_id": result.user_id, "source": result.source}

    def _to_document(self, content: str, metadata: dict[str, Any]) -> Any:
        if Document is None:
            return {"page_content": content, "metadata": metadata}
        return Document(page_content=content, metadata=metadata)

    def _split_content(self, content: str, chunk_size: int, overlap: int) -> list[str]:
        if self._text_splitter is not None:
            try:
                current_size = getattr(self._text_splitter, "_chunk_size", chunk_size)
                current_overlap = getattr(self._text_splitter, "_chunk_overlap", overlap)
                if current_size != chunk_size or current_overlap != overlap:
                    self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            except Exception:
                self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            return self._text_splitter.split_text(content)
        return split_text(content, chunk_size=chunk_size, overlap=overlap)

    def _redis_set(self, key: str, value: str) -> None:
        if self._redis:
            self._redis.set(key, value)

    def _redis_get(self, key: str) -> str:
        return self._redis.get(key) if self._redis else ""

    def upsert_script(self, user_id: str, script_id: str, content: str) -> None:
        content_key = self._redis_key("content", user_id, script_id)
        self._redis_set(content_key, content)
        if self._collection is not None:
            self._collection.insert([[user_id], [script_id], [content], [self._embed(content)]])
            self._collection.flush()
        else:
            self._legacy_memory.setdefault(user_id, {})[script_id] = content

    def add_document(self, user_id: str, doc_id: str, content: str, chunk_size: int = 1000, overlap: int = 150) -> int:
        chunks = self._split_content(content, chunk_size=chunk_size, overlap=overlap)
        documents = [self._to_document(chunk, {"user_id": user_id, "doc_id": doc_id, "chunk_index": idx + 1}) for idx, chunk in enumerate(chunks)]
        for doc in documents:
            metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            chunk_index = metadata.get("chunk_index", 1)
            chunk_content = doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
            self.upsert_script(user_id, f"{doc_id}-chunk-{chunk_index}", chunk_content)
        if self._redis:
            doc_payload = {"doc_id": doc_id, "chunks": len(chunks), "content": content, "preview": content[:800]}
            self._redis_set(self._redis_key("doc", user_id, doc_id), json.dumps(doc_payload, ensure_ascii=False))
            recent_key = self._redis_key("docs", user_id)
            recent = self._cache_get_json(recent_key) or []
            recent = [item for item in recent if item.get("doc_id") != doc_id]
            recent.insert(0, {"doc_id": doc_id, "chunks": len(chunks), "preview": content[:200]})
            self._cache_set_json(recent_key, recent[:10], ttl=86400)
        else:
            self._legacy_memory.setdefault(user_id, {})[doc_id] = content
        return len(chunks)

    def get_document_content(self, user_id: str, doc_id: str) -> str:
        if self._redis:
            payload = self._redis_get(self._redis_key("doc", user_id, doc_id))
            if payload:
                try:
                    data = json.loads(payload)
                    return data.get("content", "")
                except Exception:
                    pass
        return self._legacy_memory.get(user_id, {}).get(doc_id, "")

    def get_recent_documents(self, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        if self._redis:
            items = self._cache_get_json(self._redis_key("docs", user_id)) or []
            return list(items)[:limit]
        items = self._legacy_memory.get(user_id, {})
        return [{"doc_id": doc_id, "chunks": 1, "preview": content[:200]} for doc_id, content in list(items.items())[:limit]]

    def _search_milvus(self, user_id: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if self._collection is not None:
            try:
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                self._collection.load()
                rows = self._collection.search(data=[self._embed(query)], anns_field="embedding", param=search_params, limit=top_k, expr=f'user_id == "{user_id}"', output_fields=["user_id", "script_id", "content"])
                for hit in rows[0]:
                    entity = hit.entity
                    results.append(self._serialize_result(SearchResult(user_id=entity.get("user_id"), script_id=entity.get("script_id"), content=entity.get("content"), score=float(hit.score))))
            except Exception:
                return []
        else:
            items = self._legacy_memory.get(user_id, {})
            ranked = sorted(items.items(), key=lambda item: query.lower() not in item[1].lower())[:top_k]
            results = [self._serialize_result(SearchResult(user_id=user_id, script_id=script_id, content=content, score=1.0, source="local")) for script_id, content in ranked]
        return results

    def search(self, user_id: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        cache_key = self._redis_key("search", user_id, hashlib.sha256(query.encode("utf-8")).hexdigest(), str(top_k))
        cached = self._cache_get_json(cache_key)
        if cached is not None:
            return cached
        results = self._search_milvus(user_id, query, top_k=top_k)
        self._cache_set_json(cache_key, results, ttl=300)
        return results

    def retrieve(self, user_id: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self.search(user_id, query, top_k=top_k)

    def get_retriever_context(self, user_id: str, query: str, top_k: int = 3) -> str:
        results = self.retrieve(user_id, query, top_k=top_k)
        return "\n\n".join(item.get("content", "") for item in results if item.get("content"))

    def get_script(self, user_id: str, script_id: str) -> str:
        content_key = self._redis_key("content", user_id, script_id)
        if self._redis:
            cached = self._redis_get(content_key)
            if cached:
                return cached
        if self._collection is not None:
            try:
                expr = f'user_id == "{user_id}" and script_id == "{script_id}"'
                rows = self._collection.query(expr=expr, output_fields=["content"], limit=1)
                if rows:
                    content = rows[0].get("content", "")
                    self._redis_set(content_key, content)
                    return content
            except Exception:
                pass
        return self._legacy_memory.get(user_id, {}).get(script_id, "")

    def cache_status(self) -> dict[str, Any]:
        return {"redis": bool(self._redis), "milvus": self._collection is not None, "collection": self.collection_name, "embedding_dim": self.embedding_dim}
