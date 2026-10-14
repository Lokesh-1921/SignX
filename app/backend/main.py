"""FastAPI foundation. Model-serving endpoints become active after registration."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from models.registry import ModelRegistry

app = FastAPI(title="SignBridge API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "signbridge"}


@app.get("/models")
def models() -> list[dict[str, object]]:
    return [asdict(record) for record in ModelRegistry().list()]


@app.post("/predict")
@app.post("/translate")
@app.post("/upload")
def unavailable_model_endpoint() -> None:
    raise HTTPException(status_code=501, detail="Model serving is introduced after a validated registered checkpoint exists.")
