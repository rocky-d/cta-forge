"""FastAPI application for reporter."""

from fastapi import FastAPI

from .routes import router

app = FastAPI(title="reporter", version="0.1.0")
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
