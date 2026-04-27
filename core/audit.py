from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, path: str = "data/audit.log") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "time": datetime.utcnow().isoformat(),
            "event": event,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
