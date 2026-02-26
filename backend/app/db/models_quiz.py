import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class QuizMode(str, enum.Enum):
    multiple_choice = "multiple_choice"


class QuizStatus(str, enum.Enum):
    active    = "active"
    completed = "completed"
    abandoned = "abandoned"


class QuizSession(Base):
    """Persists the full quiz state as JSON so the MVP needs no extra tables.

    state_json schema (managed entirely by quiz_service.py):
    {
        "topic":            str,
        "questions": [
            {
                "question":      str,
                "options":       [str, ...],   # 4 items
                "correct_index": int,          # 0-based
                "explanation":   str,
                "chunk_refs":    [str, ...]    # embedding_ids used
            }
        ],
        "current_index":  int,   # next question to serve (0-based)
        "answers":        [int | null, ...],   # one slot per question
        "score":          int    # running correct count
    }
    """

    __tablename__ = "quiz_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    mode: Mapped[QuizMode] = mapped_column(
        Enum(QuizMode), default=QuizMode.multiple_choice, nullable=False
    )
    status: Mapped[QuizStatus] = mapped_column(
        Enum(QuizStatus), default=QuizStatus.active, nullable=False, index=True
    )
    state_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="{}",
        comment="Full quiz state serialised as JSON",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    workspace = relationship("Workspace", lazy="raise")

    def __repr__(self) -> str:
        return (
            f"<QuizSession id={self.id} workspace_id={self.workspace_id} "
            f"status={self.status}>"
        )