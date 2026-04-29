from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

try:
    from redis import Redis
except Exception:  # pragma: no cover
    Redis = None


@dataclass
class MemorySummary:
    worldview: str = ""
    character_state: str = ""
    plot_progress: str = ""
    pending_tasks: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "worldview": self.worldview,
            "character_state": self.character_state,
            "plot_progress": self.plot_progress,
            "pending_tasks": self.pending_tasks,
        }


class RedisChatMessageHistory:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", storage_path: str = "data/chat_history.json") -> None:
        if Redis is None or not redis_url:
            raise RuntimeError("Redis client is unavailable")
        self._client = Redis.from_url(redis_url, decode_responses=True)
        try:
            self._client.ping()
        except Exception as exc:
            raise RuntimeError(f"Redis connection failed: {exc}") from exc
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _redis_key(self, user_id: str, project_id: str) -> str:
        return f"chat:{user_id}:{project_id}"

    def _summary_key(self, user_id: str, project_id: str) -> str:
        return f"chat:summary:{user_id}:{project_id}"

    def _meta_key(self, user_id: str, project_id: str) -> str:
        return f"chat:meta:{user_id}:{project_id}"

    def _window_size(self) -> int:
        return 6

    def _summary_threshold(self) -> int:
        return 10

    def _should_store(self, message: str) -> bool:
        text = message.strip().lower()
        if not text:
            return False
        noisy = ["天气", "日期", "几号", "hello redis", "hi", "hello", "test", "debug"]
        return not any(token in text for token in noisy)

    def _load_meta(self, user_id: str, project_id: str) -> dict[str, Any]:
        raw = self._client.get(self._meta_key(user_id, project_id))
        if not raw:
            return {"role_count": 0, "script_title": "未命名剧本", "creation_stage": "对话归档", "project_type": "general"}
        try:
            meta = json.loads(raw)
            if not isinstance(meta, dict):
                raise ValueError
            return {
                "role_count": int(meta.get("role_count", 0)),
                "script_title": str(meta.get("script_title", "未命名剧本")),
                "creation_stage": str(meta.get("creation_stage", "对话归档")),
                "project_type": str(meta.get("project_type", "general")),
            }
        except Exception:
            return {"role_count": 0, "script_title": "未命名剧本", "creation_stage": "对话归档", "project_type": "general"}

    def _save_meta(self, user_id: str, project_id: str, meta: dict[str, Any]) -> None:
        self._client.set(self._meta_key(user_id, project_id), json.dumps(meta, ensure_ascii=False))

    def _load_summary(self, user_id: str, project_id: str) -> MemorySummary:
        raw = self._client.get(self._summary_key(user_id, project_id))
        if not raw:
            return MemorySummary()
        try:
            payload = json.loads(raw)
            return MemorySummary(
                worldview=str(payload.get("worldview", "")),
                character_state=str(payload.get("character_state", "")),
                plot_progress=str(payload.get("plot_progress", "")),
                pending_tasks=str(payload.get("pending_tasks", "")),
            )
        except Exception:
            return MemorySummary()

    def _save_summary(self, user_id: str, project_id: str, summary: MemorySummary) -> None:
        self._client.set(self._summary_key(user_id, project_id), json.dumps(summary.to_dict(), ensure_ascii=False))

    def _normalize_messages(self, items: list[str]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for raw in items:
            try:
                item = json.loads(raw)
            except Exception:
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            timestamp = str(item.get("timestamp") or self._now())
            if role in {"user", "assistant", "system"} and content:
                messages.append({"role": role, "content": content, "timestamp": timestamp})
        return messages

    def _compose_summary(self, history: list[dict[str, str]]) -> MemorySummary:
        user_points: list[str] = []
        assistant_points: list[str] = []
        for item in history:
            content = item.get("content", "").strip()
            if not content:
                continue
            if item.get("role") == "user":
                user_points.append(content)
            elif item.get("role") == "assistant":
                assistant_points.append(content)
        worldview = user_points[0][:120] if user_points else ""
        character_state = assistant_points[-1][:120] if assistant_points else ""
        plot_progress = "；".join(item["content"][:80] for item in history[-4:]) if history else ""
        pending_tasks = user_points[-1][:120] if user_points else ""
        return MemorySummary(worldview=worldview, character_state=character_state, plot_progress=plot_progress, pending_tasks=pending_tasks)

    def _trim_to_window(self, user_id: str, project_id: str) -> None:
        key = self._redis_key(user_id, project_id)
        items = self._client.lrange(key, 0, -1)
        if len(items) <= self._window_size():
            return
        history = self._normalize_messages(items)
        recent = history[-self._window_size():]
        summary = self._compose_summary(history[:-self._window_size()])
        self._save_summary(user_id, project_id, summary)
        self._client.delete(key)
        for item in recent:
            self._client.rpush(key, json.dumps(item, ensure_ascii=False))

    def _append_message(self, user_id: str, project_id: str, role: str, message: str, script_title: str | None = None, creation_stage: str | None = None, project_type: str | None = None) -> None:
        if not self._should_store(message):
            return
        key = self._redis_key(user_id, project_id)
        payload = {"role": role, "content": message, "timestamp": self._now()}
        items = self._client.lrange(key, 0, -1)
        if items:
            try:
                last = json.loads(items[-1])
                if last.get("role") == role and str(last.get("content", "")).strip() == message.strip():
                    return
            except Exception:
                pass
        self._client.rpush(key, json.dumps(payload, ensure_ascii=False))
        meta = self._load_meta(user_id, project_id)
        meta["role_count"] = int(meta.get("role_count", 0)) + 1
        if script_title:
            meta["script_title"] = script_title
        if creation_stage:
            meta["creation_stage"] = creation_stage
        if project_type:
            meta["project_type"] = project_type
        self._save_meta(user_id, project_id, meta)
        if meta["role_count"] >= self._summary_threshold():
            history = self._normalize_messages(self._client.lrange(key, 0, -1))
            summary = self._compose_summary(history[:-self._window_size()]) if len(history) > self._window_size() else self._compose_summary([])
            self._save_summary(user_id, project_id, summary)
            self._trim_to_window(user_id, project_id)
            meta["role_count"] = len(self._client.lrange(key, 0, -1))
            self._save_meta(user_id, project_id, meta)

    def add_user_message(self, user_id: str, message: str, project_id: str | None = None, script_title: str | None = None, creation_stage: str | None = None, project_type: str | None = None) -> None:
        self._append_message(user_id, project_id or "project_001", "user", message, script_title, creation_stage, project_type)

    def add_ai_message(self, user_id: str, message: str, project_id: str | None = None, script_title: str | None = None, creation_stage: str | None = None, project_type: str | None = None) -> None:
        self._append_message(user_id, project_id or "project_001", "assistant", message, script_title, creation_stage, project_type)

    def add_system_message(self, user_id: str, message: str, project_id: str | None = None, script_title: str | None = None, creation_stage: str | None = None, project_type: str | None = None) -> None:
        self._append_message(user_id, project_id or "project_001", "system", message, script_title, creation_stage, project_type)

    def get_messages(self, user_id: str, project_id: str | None = None) -> List[dict[str, str]]:
        project_key = project_id or "project_001"
        items = self._client.lrange(self._redis_key(user_id, project_key), 0, -1)
        return self._normalize_messages(items)

    def get_summary(self, user_id: str, project_id: str | None = None) -> dict[str, str]:
        summary = self._load_summary(user_id, project_id or "project_001")
        return summary.to_dict()

    def get_project_meta(self, user_id: str, project_id: str | None = None) -> dict[str, Any]:
        return self._load_meta(user_id, project_id or "project_001")
