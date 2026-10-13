"""FastAPI server for the verbatim RAG demo.

Run from the repo root: uv run uvicorn backend.main:app --reload
Serves /api and, when the frontend is built, the React app from frontend/dist.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from core.rag import answer, load

models = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global models
    models = load()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api")
def api(q: str) -> dict:
    if models is None:
        raise HTTPException(status_code=503, detail="models not loaded")
    return answer(q, models)


# Serve the built React app (npm run build) when it exists.
dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="static")
