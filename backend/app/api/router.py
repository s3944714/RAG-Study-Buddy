from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.indexing import router as indexing_router
from app.api.routes.workspaces import router as workspaces_router

router = APIRouter(prefix="/api/v1")

router.include_router(workspaces_router)
router.include_router(documents_router)
router.include_router(indexing_router)
router.include_router(chat_router)

# Future routers:
# from app.api.routes import quiz, flashcards
# router.include_router(quiz.router)
# router.include_router(flashcards.router)