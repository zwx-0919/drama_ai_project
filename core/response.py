from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


def success(data: Any = None, msg: str = "ok", code: int = 200) -> JSONResponse:
    return JSONResponse(status_code=200, content={"code": code, "msg": msg, "data": data})


def failure(msg: str = "error", code: int = 500, data: Any = None) -> JSONResponse:
    return JSONResponse(status_code=200, content={"code": code, "msg": msg, "data": data})
