from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ScriptRecord:
    script_id: str
    user_id: str
    title: str
    content: str
    style: str
    theme: str
    duration_min: int
    keywords: List[str]
    version: int = 1
    metadata: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class ShotRecord:
    script_id: str
    user_id: str
    content: str
