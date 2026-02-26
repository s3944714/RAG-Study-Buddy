from fastapi import FastAPI

from app.api.router import router
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="RAG Study Buddy",
    version="0.1.0",
    debug=settings.debug,
)

app.include_router(router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.env}