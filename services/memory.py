from __future__ import annotations

import json
from pathlib import Path
from typing import List

try:
    from redis import Redis
except Exception:  # pragma: no cover
    Redis = None


class RedisChatMessageHistory:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", storage_path: str = "data/chat_history.json") -> None:
        self._local_fallback_path = storage_path
        self._use_redis = Redis is not None and bool(redis_url)
        self._client = Redis.from_url(redis_url, decode_responses=True) if self._use_redis else None
        if not self._use_redis or self._client is None:
            self._path = Path(storage_path)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load_local()

    def _load_local(self) -> None:
        if not self._path.exists():
            self._store: dict[str, list[dict[str, str]]] = {}
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            self._store = {k: list(v) for k, v in payload.items()}
        except Exception:
            self._store = {}

    def _save_local(self) -> None:
        self._path.write_text(json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8")

    def _local_messages(self, user_id: str) -> list[dict[str, str]]:
        return list(self._store.get(user_id, []))

    def _redis_key(self, user_id: str) -> str:
        return f"chat:{user_id}"

    def add_user_message(self, user_id: str, message: str) -> None:
        payload = json.dumps({"role": "user", "content": message}, ensure_ascii=False)
        if self._use_redis and self._client:
            self._client.rpush(self._redis_key(user_id), payload)
        else:
            self._store.setdefault(user_id, []).append({"role": "user", "content": message})
            self._save_local()

    def add_ai_message(self, user_id: str, message: str) -> None:
        payload = json.dumps({"role": "assistant", "content": message}, ensure_ascii=False)
        if self._use_redis and self._client:
            self._client.rpush(self._redis_key(user_id), payload)
        else:
            self._store.setdefault(user_id, []).append({"role": "assistant", "content": message})
            self._save_local()

    def get_messages(self, user_id: str) -> List[dict[str, str]]:
        if self._use_redis and self._client:
            items = self._client.lrange(self._redis_key(user_id), 0, -1)
            return [json.loads(x) for x in items]
        return self._local_messages(user_id)

    def add_system_message(self, user_id: str, message: str) -> None:
        payload = json.dumps({"role": "system", "content": message}, ensure_ascii=False)
        if self._use_redis and self._client:
            self._client.rpush(self._redis_key(user_id), payload)
        else:
            self._store.setdefault(user_id, []).append({"role": "system", "content": message})
            self._save_local()
