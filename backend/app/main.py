"""Architect SaaS API — 2D floor plan to 3D BIM reconstruction service."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import store
from .api.routes import router
from .config import settings
from .security import SecurityMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    yield
    store.close()


app = FastAPI(
    title="Architect SaaS",
    version="0.1.0",
    lifespan=lifespan,
    # API schema stays available in dev; disable docs in production via env.
)

app.add_middleware(SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
