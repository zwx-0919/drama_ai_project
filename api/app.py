from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.exceptions import AppError, app_exception_handler, generic_exception_handler
from routers import health, script, chat


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(AppError, app_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    app.include_router(health, prefix=settings.api_prefix)
    app.include_router(script, prefix=f"{settings.api_prefix}/script")
    app.include_router(chat, prefix=f"{settings.api_prefix}/chat")
    return app
