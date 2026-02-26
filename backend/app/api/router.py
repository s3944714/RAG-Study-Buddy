from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")

# Sub-routers will be registered here as modules are added, e.g.:
# from app.api.endpoints import ingest, query
# router.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
# router.include_router(query.router,  prefix="/query",  tags=["retrieval"])