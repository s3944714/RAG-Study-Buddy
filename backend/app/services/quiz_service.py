"""
Quiz service — generates questions, manages session state in DB JSON.

State lifecycle:
    start()  → creates QuizSession row, generates all questions up front,
               stores them in state_json, returns first question.
    next()   → increments current_index, returns next question or summary.
    answer() → records the chosen option, checks correctness, advances index.

All question generation is done at start-time so subsequent calls are
instant (no LLM round-trip per question).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Workspace
from app.db.models_quiz import QuizMode, QuizSession, QuizStatus
from app.retrieval.retriever import RetrievedChunk, retrieve
from app.services.embeddings_client import EmbeddingsClient, get_embeddings_client
from app.services.llm_client import LLMClient, get_llm_client
from app.services.vector_store import VectorStoreClient, get_vector_store_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

QUIZ_SYSTEM_PROMPT: str = (
    "You are a quiz generator for a study application. "
    "Given source passages, return ONLY a valid JSON array — no prose, no fences. "
    "Each element must have exactly these keys:\n"
    '  "question"      : string — the question text\n'
    '  "options"       : array of exactly 4 strings\n'
    '  "correct_index" : integer 0–3 (index into options)\n'
    '  "explanation"   : string — one-sentence explanation of the correct answer\n'
    "Base every question strictly on the provided passages. Do NOT invent facts."
)

_QUIZ_USER_TEMPLATE: str = """\
Generate exactly {n} multiple-choice questions about: {topic}

Source passages:
{context}

Return ONLY a JSON array. Example:
[{{"question":"What is X?","options":["A","B","C","D"],"correct_index":0,"explanation":"X is A because..."}}]
"""

# ---------------------------------------------------------------------------
# Pydantic-free internal state helpers (plain dicts kept for speed)
# ---------------------------------------------------------------------------

def _load_state(session_row: QuizSession) -> dict[str, Any]:
    return json.loads(session_row.state_json)


def _save_state(session_row: QuizSession, state: dict[str, Any]) -> None:
    session_row.state_json = json.dumps(state)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        heading = f" | {c.heading}" if c.heading else ""
        parts.append(f"[{i}] Doc {c.doc_id}, p.{c.page}{heading}\n{c.text[:500]}")
    return "\n\n".join(parts) if parts else "(No relevant passages found.)"


# ---------------------------------------------------------------------------
# LLM JSON parsing + fallback
# ---------------------------------------------------------------------------

def _parse_questions(raw: str) -> list[dict[str, Any]]:
    """Extract a JSON array of question objects from LLM output."""
    text = raw.strip()

    for candidate in (text, re.search(r"\[.*\]", text, re.DOTALL)):
        snippet = candidate if isinstance(candidate, str) else (
            candidate.group() if candidate else None
        )
        if snippet is None:
            continue
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def _validate_question(q: Any) -> dict[str, Any] | None:
    """Return cleaned question dict or None if malformed."""
    if not isinstance(q, dict):
        return None
    question = str(q.get("question", "")).strip()
    options  = q.get("options", [])
    try:
        correct_index = int(q.get("correct_index", -1))
    except (TypeError, ValueError):
        return None
    explanation = str(q.get("explanation", "")).strip()

    if (
        not question
        or not isinstance(options, list)
        or len(options) != 4
        or not all(isinstance(o, str) and o.strip() for o in options)
        or correct_index not in range(4)
    ):
        return None

    return {
        "question":      question,
        "options":       [o.strip() for o in options],
        "correct_index": correct_index,
        "explanation":   explanation,
    }


def _fallback_questions(
    chunks: list[RetrievedChunk],
    n: int,
) -> list[dict[str, Any]]:
    """Derive stub questions directly from chunks (used when LLM JSON unparseable)."""
    questions = []
    for chunk in chunks[:n]:
        front = (
            f"According to the passage, what is '{chunk.heading}'?"
            if chunk.heading
            else f"What does the passage from Doc {chunk.doc_id} (p.{chunk.page}) describe?"
        )
        correct = chunk.text[:120].strip() or "See source passage."
        questions.append({
            "question":      front,
            "options":       [correct, "None of the above", "Not mentioned", "Unclear"],
            "correct_index": 0,
            "explanation":   f"The passage states: {correct[:100]}",
        })
    return questions


# ---------------------------------------------------------------------------
# Response builders (plain dicts → serialisable, no FastAPI dep here)
# ---------------------------------------------------------------------------

def _question_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Return the current question for the client (no answer/explanation)."""
    idx = state["current_index"]
    q   = state["questions"][idx]
    return {
        "session_id":       None,   # filled in by caller
        "question_number":  idx + 1,
        "total_questions":  len(state["questions"]),
        "question":         q["question"],
        "options":          q["options"],
        "status":           "active",
    }


def _summary_payload(state: dict[str, Any], session_id: int) -> dict[str, Any]:
    total   = len(state["questions"])
    correct = state["score"]
    return {
        "session_id":    session_id,
        "status":        "completed",
        "score":         correct,
        "total":         total,
        "percentage":    round(correct / total * 100, 1) if total else 0.0,
        "answers":       state["answers"],
    }


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

async def start_quiz(
    *,
    db: AsyncSession,
    workspace_id: int,
    topic: str,
    num_questions: int = 5,
    mode: QuizMode = QuizMode.multiple_choice,
    embeddings_client: EmbeddingsClient | None = None,
    vector_store: VectorStoreClient | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Create a QuizSession, generate all questions, return the first one."""
    if not topic.strip():
        raise ValueError("topic must not be empty")
    num_questions = max(1, min(20, num_questions))

    # Validate workspace exists
    ws = await db.get(Workspace, workspace_id)
    if ws is None:
        raise ValueError(f"Workspace {workspace_id} not found")

    emb = embeddings_client or get_embeddings_client()
    vs  = vector_store      or get_vector_store_client()
    llm = llm_client        or get_llm_client()

    # 1. Retrieve context
    chunks = await retrieve(
        workspace_id, topic,
        top_k=min(num_questions * 2, 20),
        embeddings_client=emb,
        vector_store=vs,
    )
    context = _build_context(chunks)

    # 2. Generate questions via LLM
    user_prompt = _QUIZ_USER_TEMPLATE.format(
        n=num_questions, topic=topic.strip(), context=context
    )
    raw = await llm.generate(user_prompt, system_prompt=QUIZ_SYSTEM_PROMPT)

    raw_questions = _parse_questions(raw)
    if not raw_questions:
        logger.warning("LLM returned unparseable quiz JSON — using fallback for ws=%d", workspace_id)
        raw_questions = _fallback_questions(chunks, num_questions)

    validated = [_validate_question(q) for q in raw_questions]
    questions  = [q for q in validated if q is not None][:num_questions]

    if not questions:
        raise ValueError(
            "Could not generate any valid questions — upload more material about this topic."
        )

    # 3. Persist session
    state: dict[str, Any] = {
        "topic":         topic.strip(),
        "questions":     questions,
        "current_index": 0,
        "answers":       [None] * len(questions),
        "score":         0,
    }
    session_row = QuizSession(
        workspace_id=workspace_id,
        mode=mode,
        status=QuizStatus.active,
    )
    _save_state(session_row, state)
    db.add(session_row)
    await db.commit()
    await db.refresh(session_row)

    payload = _question_payload(state)
    payload["session_id"] = session_row.id
    logger.info(
        "Started quiz session_id=%d ws=%d topic=%r questions=%d",
        session_row.id, workspace_id, topic[:40], len(questions),
    )
    return payload


async def next_question(
    *,
    db: AsyncSession,
    session_id: int,
) -> dict[str, Any]:
    """Advance to the next unanswered question or return the completion summary."""
    session_row = await db.get(QuizSession, session_id)
    if session_row is None:
        raise ValueError(f"Quiz session {session_id} not found")
    if session_row.status != QuizStatus.active:
        raise ValueError(f"Quiz session {session_id} is already {session_row.status.value}")

    state = _load_state(session_row)
    idx   = state["current_index"]
    total = len(state["questions"])

    # Already past the end → return summary
    if idx >= total:
        session_row.status = QuizStatus.completed
        await db.commit()
        return _summary_payload(state, session_id)

    payload = _question_payload(state)
    payload["session_id"] = session_id
    return payload


async def submit_answer(
    *,
    db: AsyncSession,
    session_id: int,
    chosen_index: int,
) -> dict[str, Any]:
    """Record the player's answer and advance the index.

    Returns feedback for the current question plus the next question
    (or a completion summary when all questions are exhausted).
    """
    session_row = await db.get(QuizSession, session_id)
    if session_row is None:
        raise ValueError(f"Quiz session {session_id} not found")
    if session_row.status != QuizStatus.active:
        raise ValueError(f"Quiz session {session_id} is already {session_row.status.value}")

    state = _load_state(session_row)
    idx   = state["current_index"]
    total = len(state["questions"])

    if idx >= total:
        raise ValueError("No more questions — session should be completed.")
    if chosen_index not in range(4):
        raise ValueError("chosen_index must be 0–3")

    q       = state["questions"][idx]
    correct = chosen_index == q["correct_index"]
    if correct:
        state["score"] += 1
    state["answers"][idx] = chosen_index
    state["current_index"] = idx + 1

    feedback: dict[str, Any] = {
        "session_id":      session_id,
        "question_number": idx + 1,
        "chosen_index":    chosen_index,
        "correct_index":   q["correct_index"],
        "is_correct":      correct,
        "explanation":     q["explanation"],
        "score_so_far":    state["score"],
    }

    # Persist updated state
    _save_state(session_row, state)

    # Advance to next question or mark completed
    if state["current_index"] >= total:
        session_row.status = QuizStatus.completed
        await db.commit()
        feedback["next"] = _summary_payload(state, session_id)
    else:
        await db.commit()
        next_payload = _question_payload(state)
        next_payload["session_id"] = session_id
        feedback["next"] = next_payload

    return feedback