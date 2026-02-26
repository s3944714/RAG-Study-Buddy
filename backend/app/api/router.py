from fastapi import APIRouter

from app.api.routes.workspaces import router as workspaces_router
from app.api.routes.documents import router as documents_router

router = APIRouter(prefix="/api/v1")

router.include_router(workspaces_router)
router.include_router(documents_router)

# Future routers:
# from app.api.routes import chat, quiz, flashcards
# router.include_router(chat.router)