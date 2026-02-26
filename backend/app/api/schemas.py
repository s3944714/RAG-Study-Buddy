from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, examples=["Physics 101"])


class WorkspaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    workspace_id: int
    filename: str
    storage_path: str
    file_hash: str
    status: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    document_id: int
    filename: str
    status: str
    message: str = ""


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    page: int | None = None
    heading: str | None = None
    excerpt: str = Field(..., max_length=400)


class ChatRequest(BaseModel):
    workspace_id: int
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description='List of {"role": "user"|"assistant", "content": "..."} turns',
    )


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model: str = ""


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

class QuizRequest(BaseModel):
    workspace_id: int
    document_ids: list[int] = Field(
        default_factory=list,
        description="Empty list means quiz across the whole workspace",
    )
    num_questions: int = Field(default=5, ge=1, le=20)
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(..., min_length=2, max_length=6)
    correct_index: int = Field(..., description="0-based index into options")
    explanation: str = ""


class QuizResponse(BaseModel):
    workspace_id: int
    questions: list[QuizQuestion]
    model: str = ""


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------

class Flashcard(BaseModel):
    front: str = Field(..., description="Question or term")
    back: str = Field(..., description="Answer or definition")
    source_chunk_id: int | None = None


class FlashcardSetResponse(BaseModel):
    workspace_id: int
    document_ids: list[int] = Field(default_factory=list)
    flashcards: list[Flashcard]
    model: str = ""




