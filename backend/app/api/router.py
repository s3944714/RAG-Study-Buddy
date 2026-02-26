from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.export import router as export_router
from app.api.routes.flashcards import router as flashcards_router
from app.api.routes.indexing import router as indexing_router
from app.api.routes.quiz import router as quiz_router
from app.api.routes.workspaces import router as workspaces_router

router = APIRouter(prefix="/api/v1")

router.include_router(workspaces_router)
router.include_router(documents_router)
router.include_router(indexing_router)
router.include_router(chat_router)
router.include_router(flashcards_router)
router.include_router(quiz_router)
router.include_router(export_router)