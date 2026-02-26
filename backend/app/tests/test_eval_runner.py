from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.eval.eval_runner import (
    aggregate,
    compute_citation_coverage,
    compute_keyword_hit_rate,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_unsupported_claims_flag,
    evaluate_case,
    run_eval,
)
from app.retrieval.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# compute_recall_at_k
# ---------------------------------------------------------------------------

def test_recall_hit() -> None:
    assert compute_recall_at_k([1, 2], [2, 3, 4]) == 1.0


def test_recall_miss() -> None:
    assert compute_recall_at_k([1, 2], [3, 4, 5]) == 0.0


def test_recall_empty_expected_is_perfect() -> None:
    assert compute_recall_at_k([], [1, 2, 3]) == 1.0


def test_recall_empty_retrieved_is_zero() -> None:
    assert compute_recall_at_k([1], []) == 0.0


def test_recall_exact_match() -> None:
    assert compute_recall_at_k([5], [5]) == 1.0


# ---------------------------------------------------------------------------
# compute_precision_at_k
# ---------------------------------------------------------------------------

def test_precision_all_correct() -> None:
    assert compute_precision_at_k([1, 2], [1, 2]) == 1.0


def test_precision_none_correct() -> None:
    assert compute_precision_at_k([1, 2], [3, 4]) == 0.0


def test_precision_partial() -> None:
    result = compute_precision_at_k([1], [1, 2, 3, 4])
    assert result == pytest.approx(0.25)


def test_precision_empty_retrieved_is_zero() -> None:
    assert compute_precision_at_k([1], []) == 0.0


def test_precision_empty_expected() -> None:
    # Nothing expected → nothing can match → 0.0
    assert compute_precision_at_k([], [1, 2]) == 0.0


# ---------------------------------------------------------------------------
# compute_citation_coverage
# ---------------------------------------------------------------------------

def test_citation_coverage_with_citations() -> None:
    assert compute_citation_coverage([{"document_id": 1}]) == 1.0


def test_citation_coverage_empty() -> None:
    assert compute_citation_coverage([]) == 0.0


def test_citation_coverage_multiple() -> None:
    assert compute_citation_coverage([{"a": 1}, {"b": 2}]) == 1.0


# ---------------------------------------------------------------------------
# compute_unsupported_claims_flag
# ---------------------------------------------------------------------------

def test_unsupported_no_citations_with_marker() -> None:
    assert compute_unsupported_claims_flag(
        "Research has shown that cells divide rapidly.", []
    ) is True


def test_unsupported_with_citations_suppresses_flag() -> None:
    assert compute_unsupported_claims_flag(
        "Research has shown that cells divide rapidly.",
        [{"document_id": 1}],
    ) is False


def test_unsupported_clean_answer_no_flag() -> None:
    assert compute_unsupported_claims_flag(
        "The passage states that the nucleus contains DNA.", []
    ) is False


def test_unsupported_marker_it_is_known() -> None:
    assert compute_unsupported_claims_flag(
        "It is well known that water boils at 100°C.", []
    ) is True


def test_unsupported_marker_studies_show() -> None:
    assert compute_unsupported_claims_flag(
        "Studies show a correlation between X and Y.", []
    ) is True


def test_unsupported_empty_answer() -> None:
    assert compute_unsupported_claims_flag("", []) is False


# ---------------------------------------------------------------------------
# compute_keyword_hit_rate
# ---------------------------------------------------------------------------

def test_keyword_all_present() -> None:
    assert compute_keyword_hit_rate("ATP is the energy currency.", ["ATP", "energy"]) == 1.0


def test_keyword_none_present() -> None:
    assert compute_keyword_hit_rate("Completely unrelated text.", ["DNA", "RNA"]) == 0.0


def test_keyword_partial() -> None:
    rate = compute_keyword_hit_rate("The nucleus contains DNA.", ["DNA", "RNA", "protein"])
    assert rate == pytest.approx(1 / 3)


def test_keyword_empty_keywords_is_perfect() -> None:
    assert compute_keyword_hit_rate("anything", []) == 1.0


def test_keyword_case_insensitive() -> None:
    assert compute_keyword_hit_rate("Mitochondria produce atp.", ["ATP", "mitochondria"]) == 1.0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def test_aggregate_empty_returns_empty_dict() -> None:
    assert aggregate([]) == {}


def test_aggregate_computes_means() -> None:
    results = [
        {"recall_at_k": 1.0, "precision_at_k": 0.5, "citation_coverage": 1.0,
         "keyword_hit_rate": 0.8, "unsupported_claims_flag": False,
         "chunks_retrieved": 3, "num_citations": 2},
        {"recall_at_k": 0.0, "precision_at_k": 0.0, "citation_coverage": 0.0,
         "keyword_hit_rate": 0.4, "unsupported_claims_flag": True,
         "chunks_retrieved": 1, "num_citations": 0},
    ]
    summary = aggregate(results)
    assert summary["mean_recall_at_k"]       == pytest.approx(0.5)
    assert summary["mean_precision_at_k"]    == pytest.approx(0.25)
    assert summary["mean_citation_coverage"] == pytest.approx(0.5)
    assert summary["unsupported_claims_flagged"] == 1
    assert summary["num_cases"] == 2


def test_aggregate_flag_rate() -> None:
    results = [
        {"recall_at_k": 1.0, "precision_at_k": 1.0, "citation_coverage": 1.0,
         "keyword_hit_rate": 1.0, "unsupported_claims_flag": True,
         "chunks_retrieved": 5, "num_citations": 0},
        {"recall_at_k": 1.0, "precision_at_k": 1.0, "citation_coverage": 1.0,
         "keyword_hit_rate": 1.0, "unsupported_claims_flag": True,
         "chunks_retrieved": 5, "num_citations": 0},
        {"recall_at_k": 1.0, "precision_at_k": 1.0, "citation_coverage": 1.0,
         "keyword_hit_rate": 1.0, "unsupported_claims_flag": False,
         "chunks_retrieved": 5, "num_citations": 1},
    ]
    summary = aggregate(results)
    assert summary["unsupported_claims_flag_rate"] == pytest.approx(2 / 3, rel=1e-3)


# ---------------------------------------------------------------------------
# evaluate_case — unit (mocked retrieve + run_chat)
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        embedding_id=f"emb-{doc_id}", text="Some text.",
        doc_id=doc_id, page=1, heading="", score=0.9, filename="f.pdf",
    )


@pytest.mark.asyncio
async def test_evaluate_case_happy_path() -> None:
    from app.api.schemas import ChatResponse, Citation

    mock_citation = Citation(
        document_id=1, filename="f.pdf", chunk_index=0,
        page=1, heading=None, excerpt="excerpt",
    )
    mock_response = ChatResponse(
        answer="The nucleus contains DNA.", citations=[mock_citation], model="dummy"
    )

    with (
        patch("app.eval.eval_runner.retrieve", new_callable=AsyncMock,
              return_value=[_make_chunk(1), _make_chunk(2)]),
        patch("app.eval.eval_runner.run_chat", new_callable=AsyncMock,
              return_value=mock_response),
    ):
        result = await evaluate_case(
            {"id": "t1", "query": "What is the nucleus?",
             "expected_doc_ids": [1], "reference_keywords": ["DNA", "nucleus"]},
            workspace_id=1, top_k=5,
            embeddings_client=MagicMock(),
            vector_store=MagicMock(),
            llm_client=MagicMock(),
        )

    assert result["case_id"]          == "t1"
    assert result["recall_at_k"]      == 1.0
    assert result["citation_coverage"] == 1.0
    assert result["keyword_hit_rate"]  > 0.0
    assert result["chunks_retrieved"] == 2


@pytest.mark.asyncio
async def test_evaluate_case_retrieval_error_graceful() -> None:
    """Retrieval failure must not crash — returns zeroed metrics."""
    from app.api.schemas import ChatResponse
    mock_response = ChatResponse(answer="fallback", citations=[], model="dummy")

    with (
        patch("app.eval.eval_runner.retrieve",
              new_callable=AsyncMock, side_effect=RuntimeError("store down")),
        patch("app.eval.eval_runner.run_chat",
              new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await evaluate_case(
            {"id": "err", "query": "q", "expected_doc_ids": [1], "reference_keywords": []},
            workspace_id=1, top_k=5,
            embeddings_client=MagicMock(),
            vector_store=MagicMock(),
            llm_client=MagicMock(),
        )

    assert result["recall_at_k"]      == 0.0
    assert result["chunks_retrieved"] == 0


@pytest.mark.asyncio
async def test_evaluate_case_chat_error_graceful() -> None:
    """Chat failure must not crash — citations empty, unsupported_claims False."""
    with (
        patch("app.eval.eval_runner.retrieve",
              new_callable=AsyncMock, return_value=[_make_chunk(1)]),
        patch("app.eval.eval_runner.run_chat",
              new_callable=AsyncMock, side_effect=RuntimeError("LLM down")),
    ):
        result = await evaluate_case(
            {"id": "err2", "query": "q", "expected_doc_ids": [1], "reference_keywords": []},
            workspace_id=1, top_k=5,
            embeddings_client=MagicMock(),
            vector_store=MagicMock(),
            llm_client=MagicMock(),
        )

    assert result["citation_coverage"] == 0.0
    assert result["model_used"]         == "error"


# ---------------------------------------------------------------------------
# run_eval — integration (mocked I/O)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_eval_writes_report(tmp_path: Path) -> None:
    from app.api.schemas import ChatResponse, Citation

    mock_citation = Citation(
        document_id=1, filename="f.pdf", chunk_index=0,
        page=1, heading=None, excerpt="text",
    )
    mock_response = ChatResponse(
        answer="The answer.", citations=[mock_citation], model="dummy"
    )

    with (
        patch("app.eval.eval_runner.retrieve",
              new_callable=AsyncMock, return_value=[_make_chunk(1)]),
        patch("app.eval.eval_runner.run_chat",
              new_callable=AsyncMock, return_value=mock_response),
    ):
        report = await run_eval(
            eval_set_path=Path(__file__).parent.parent / "eval" / "sample_eval_set.json",
            report_dir=tmp_path,
            top_k=3,
            workspace_id_override=1,
        )

    # report.json must exist
    assert (tmp_path / "report.json").exists()

    saved = json.loads((tmp_path / "report.json").read_text())
    assert "summary" in saved
    assert "results" in saved
    assert saved["summary"]["num_cases"] > 0


@pytest.mark.asyncio
async def test_run_eval_missing_eval_set_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await run_eval(
            eval_set_path=tmp_path / "nonexistent.json",
            report_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_run_eval_report_contains_all_summary_keys(tmp_path: Path) -> None:
    from app.api.schemas import ChatResponse
    mock_response = ChatResponse(answer="answer", citations=[], model="dummy")

    with (
        patch("app.eval.eval_runner.retrieve",
              new_callable=AsyncMock, return_value=[]),
        patch("app.eval.eval_runner.run_chat",
              new_callable=AsyncMock, return_value=mock_response),
    ):
        report = await run_eval(
            eval_set_path=Path(__file__).parent.parent / "eval" / "sample_eval_set.json",
            report_dir=tmp_path,
            top_k=3,
            workspace_id_override=1,
        )

    expected_keys = {
        "num_cases", "mean_recall_at_k", "mean_precision_at_k",
        "mean_citation_coverage", "mean_keyword_hit_rate",
        "unsupported_claims_flagged", "unsupported_claims_flag_rate",
        "mean_chunks_retrieved", "mean_num_citations",
    }
    assert expected_keys.issubset(report["summary"].keys())