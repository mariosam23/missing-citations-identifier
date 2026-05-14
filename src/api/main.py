"""FastAPI application factory and ASGI entry point.

Uvicorn target: ``uvicorn src.api.main:app --reload`` (run from repo root with
``PYTHONPATH=src``) or simply ``uvicorn api.main:app --reload``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, paper, recommend


def create_app() -> FastAPI:
    app = FastAPI(
        title="Citation Recommender",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(vscode-webview://.*|http://localhost(:\d+)?)$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(recommend.router)
    app.include_router(paper.router)

    return app


app = create_app()
