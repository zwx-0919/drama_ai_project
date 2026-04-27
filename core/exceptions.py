from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(self, message: str, code: int = 400):
        self.message = message
        self.code = code
        super().__init__(message)


async def app_exception_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=200, content={"code": exc.code, "msg": exc.message, "data": None})


async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=200, content={"code": 500, "msg": f"系统异常: {exc}", "data": None})
