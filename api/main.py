from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import auth, token_store
from config import settings
from workers.processing_service import processing_service

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.GMAIL_OAUTH_REFRESH_TOKEN:
        token_store.set_tokens(refresh_token=settings.GMAIL_OAUTH_REFRESH_TOKEN)
    processing_service.start()
    yield
    processing_service.stop()


app = FastAPI(title="Butlr API", lifespan=lifespan)
app.include_router(auth.router)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/result")
def get_result() -> dict:
    snapshot = processing_service.get_snapshot()

    if snapshot.result is None and snapshot.last_error is None:
        snapshot = processing_service.run_once()

    return {
        "result": snapshot.result.model_dump() if snapshot.result else None,
        "last_run_at": snapshot.last_run_at,
        "next_poll_at": snapshot.next_poll_at,
        "last_error": snapshot.last_error,
        "processed_email_count": snapshot.processed_email_count,
        "poll_interval_seconds": max(settings.GMAIL_POLL_INTERVAL_SECONDS, 15),
    }


@app.post("/run")
def run() -> dict:
    snapshot = processing_service.run_once(force=True)
    if snapshot.result is None:
        raise HTTPException(status_code=500, detail=snapshot.last_error or "Run failed")

    return {
        "result": snapshot.result.model_dump(),
        "last_run_at": snapshot.last_run_at,
        "next_poll_at": snapshot.next_poll_at,
        "last_error": snapshot.last_error,
        "processed_email_count": snapshot.processed_email_count,
        "poll_interval_seconds": max(settings.GMAIL_POLL_INTERVAL_SECONDS, 15),
    }
