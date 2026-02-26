"""
RAG Study Buddy — Evaluation Runner
====================================

Measures retrieval quality and answer groundedness against a labelled eval set.

Metrics
-------
recall@k
    Fraction of eval cases where at least one expected_doc_id appears in the
    top-k retrieved chunks.  Range [0, 1]; higher is better.

precision@k
    Fraction of retrieved doc_ids (across top-k chunks) that were expected.
    Range [0, 1]; higher is better.

citation_coverage
    Fraction of answers that contain at least one citation object.
    A citation-free answer is a red flag for a grounded RAG system.

unsupported_claims_flag (heuristic proxy)
    Flags an answer if it contains phrases associated with hallucination or
    reasoning beyond the retrieved context ("it is known that", "generally",
    "typically", "in most cases", "studies show", "research suggests", etc.)
    combined with zero citations.  This is NOT a semantic entailment check —
    it is a cheap surface heuristic that catches obvious failures.

keyword_hit_rate
    Fraction of reference_keywords (from the eval set) present in the answer
    text (case-insensitive).  Proxy for answer completeness.

How to run
----------
From the repo root (WSL / bash):

    cd backend
    poetry run python -m app.eval.eval_runner

Optional env vars:

    EVAL_SET_PATH      path to eval JSON  (default: app/eval/sample_eval_set.json)
    EVAL_REPORT_DIR    directory for output (default: eval_reports/)
    EVAL_TOP_K         retrieval top-k     (default: 5)
    EVAL_WORKSPACE_ID  override workspace  (default: from eval set JSON)

    # Provider env vars used at runtime (same as main app):
    EMBEDDINGS_PROVIDER=dummy   # use DummyEmbeddingsClient for offline eval
    LLM_PROVIDER=dummy          # use DummyLLMClient for offline eval
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — allow running as  python -m app.eval.eval_runner
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[3]  # backend/
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.retrieval.retriever import retrieve
from app.services.chat_service import run_chat
from app.services.embeddings_client import get_embeddings_client
from app.services.vector_store import get_vector_store_client
from app.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

_DEFAULT_EVAL_SET = Path(__file__).parent / "sample_eval_set.json"
_DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[3] / "eval_reports"
_DEFAULT_TOP_K = 5

# Phrases that suggest the LLM is drawing on knowledge outside the context.
_HALLUCINATION_MARKERS: list[str] = [
    r"it is (well )?known that",
    r"(generally|typically|usually|commonly) (speaking,? )?",
    r"in most cases",
    r"studies (have shown|show|suggest)",
    r"research (has shown|shows|suggests|indicates)",
    r"according to (experts|scientists|researchers)",
    r"historically",
    r"it is widely (accepted|believed)",
    r"as (everyone|most people) know",
]
_HALLUCINATION_RE = re.compile(
    "|".join(_HALLUCINATION_MARKERS), re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Metric helpers (pure functions — easy to unit-test)
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    expected_doc_ids: list[int],
    retrieved_doc_ids: list[int],
) -> float:
    """1.0 if any expected doc appears in retrieved_doc_ids, else 0.0."""
    if not expected_doc_ids:
        return 1.0  # nothing expected → trivially satisfied
    return 1.0 if set(expected_doc_ids) & set(retrieved_doc_ids) else 0.0


def compute_precision_at_k(
    expected_doc_ids: list[int],
    retrieved_doc_ids: list[int],
) -> float:
    """Fraction of retrieved docs that were expected."""
    if not retrieved_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    hits = sum(1 for d in retrieved_doc_ids if d in expected)
    return hits / len(retrieved_doc_ids)


def compute_citation_coverage(citations: list[dict]) -> float:
    """1.0 if at least one citation is present, else 0.0."""
    return 1.0 if citations else 0.0


def compute_unsupported_claims_flag(answer: str, citations: list[dict]) -> bool:
    """Heuristic: True (flagged) when hallucination markers found AND no citations."""
    if citations:
        return False  # cited answers are considered grounded
    return bool(_HALLUCINATION_RE.search(answer))


def compute_keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    """Fraction of reference keywords (case-insensitive) present in the answer."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# Per-case evaluator
# ---------------------------------------------------------------------------

async def evaluate_case(
    case: dict,
    *,
    workspace_id: int,
    top_k: int,
    embeddings_client,
    vector_store,
    llm_client,
) -> dict:
    """Run retrieval + chat for one eval case and return a result dict."""
    case_id   = case["id"]
    query     = case["query"]
    expected  = case.get("expected_doc_ids", [])
    keywords  = case.get("reference_keywords", [])

    logger.info("  Evaluating [%s]: %r", case_id, query[:70])

    # --- Retrieval ---
    try:
        chunks = await retrieve(
            workspace_id,
            query,
            top_k=top_k,
            embeddings_client=embeddings_client,
            vector_store=vector_store,
        )
        retrieved_doc_ids = [c.doc_id for c in chunks]
    except Exception as exc:
        logger.error("  Retrieval error for [%s]: %s", case_id, exc)
        retrieved_doc_ids = []
        chunks = []

    recall    = compute_recall_at_k(expected, retrieved_doc_ids)
    precision = compute_precision_at_k(expected, retrieved_doc_ids)

    # --- Chat (answer + citations) ---
    try:
        response = await run_chat(
            workspace_id=workspace_id,
            question=query,
            top_k=top_k,
            embeddings_client=embeddings_client,
            vector_store=vector_store,
            llm_client=llm_client,
        )
        answer      = response.answer
        citations   = [c.model_dump() for c in response.citations]
        model_used  = response.model
    except Exception as exc:
        logger.error("  Chat error for [%s]: %s", case_id, exc)
        answer     = ""
        citations  = []
        model_used = "error"

    citation_cov  = compute_citation_coverage(citations)
    unsupported   = compute_unsupported_claims_flag(answer, citations)
    kw_hit_rate   = compute_keyword_hit_rate(answer, keywords)

    return {
        "case_id":                 case_id,
        "query":                   query,
        "expected_doc_ids":        expected,
        "retrieved_doc_ids":       retrieved_doc_ids,
        "chunks_retrieved":        len(chunks),
        "recall_at_k":             recall,
        "precision_at_k":          precision,
        "citation_coverage":       citation_cov,
        "unsupported_claims_flag": unsupported,
        "keyword_hit_rate":        round(kw_hit_rate, 3),
        "num_citations":           len(citations),
        "model_used":              model_used,
        "answer_preview":          answer[:200],
        "notes":                   case.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(results: list[dict]) -> dict:
    """Compute mean metrics across all eval cases."""
    if not results:
        return {}

    n = len(results)

    def mean(key: str) -> float:
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    flagged = sum(1 for r in results if r.get("unsupported_claims_flag"))

    return {
        "num_cases":                    n,
        "mean_recall_at_k":             mean("recall_at_k"),
        "mean_precision_at_k":          mean("precision_at_k"),
        "mean_citation_coverage":       mean("citation_coverage"),
        "mean_keyword_hit_rate":        mean("keyword_hit_rate"),
        "unsupported_claims_flagged":   flagged,
        "unsupported_claims_flag_rate": round(flagged / n, 4),
        "mean_chunks_retrieved":        mean("chunks_retrieved"),
        "mean_num_citations":           mean("num_citations"),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_eval(
    eval_set_path: Path | None = None,
    report_dir: Path | None = None,
    top_k: int | None = None,
    workspace_id_override: int | None = None,
) -> dict:
    """Load eval set, evaluate all cases, write JSON report, return report dict."""
    eval_set_path = eval_set_path or Path(
        os.environ.get("EVAL_SET_PATH", str(_DEFAULT_EVAL_SET))
    )
    report_dir = report_dir or Path(
        os.environ.get("EVAL_REPORT_DIR", str(_DEFAULT_REPORT_DIR))
    )
    top_k = top_k or int(os.environ.get("EVAL_TOP_K", str(_DEFAULT_TOP_K)))

    # Load eval set
    if not eval_set_path.exists():
        raise FileNotFoundError(f"Eval set not found: {eval_set_path}")
    with eval_set_path.open() as f:
        eval_set = json.load(f)

    workspace_id = workspace_id_override or int(
        os.environ.get("EVAL_WORKSPACE_ID", str(eval_set.get("workspace_id", 1)))
    )
    cases = eval_set.get("eval_cases", [])

    logger.info(
        "Starting eval: %d cases | workspace_id=%d | top_k=%d",
        len(cases), workspace_id, top_k,
    )

    # Build shared clients once
    emb = get_embeddings_client()
    vs  = get_vector_store_client()
    llm = get_llm_client()

    # Evaluate all cases
    results: list[dict] = []
    for case in cases:
        result = await evaluate_case(
            case,
            workspace_id=workspace_id,
            top_k=top_k,
            embeddings_client=emb,
            vector_store=vs,
            llm_client=llm,
        )
        results.append(result)

    summary = aggregate(results)

    report = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "eval_set_path":   str(eval_set_path),
        "eval_set_version": eval_set.get("version", "unknown"),
        "workspace_id":    workspace_id,
        "top_k":           top_k,
        "summary":         summary,
        "results":         results,
    }

    # Write report
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path  = report_dir / "report.json"
    archive_path = report_dir / f"report_{ts}.json"

    for path in (report_path, archive_path):
        with path.open("w") as f:
            json.dump(report, f, indent=2)
        logger.info("Report written → %s", path)

    # Print summary to stdout
    logger.info("=" * 60)
    logger.info("EVAL SUMMARY")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info("  %-40s %s", k, v)
    logger.info("=" * 60)

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_eval())