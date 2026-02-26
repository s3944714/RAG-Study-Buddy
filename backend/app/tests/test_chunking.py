import pytest

from app.ingestion.chunking import chunk_pages, _best_heading
from app.ingestion.pdf_extractor import PageText


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pages(*texts: str) -> list[PageText]:
    return [PageText(page_number=i + 1, text=t) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_returns_list_of_chunk_candidates() -> None:
    pages = _pages("Hello world " * 50)
    chunks = chunk_pages(pages)
    assert isinstance(chunks, list)
    assert all(isinstance(c, dict) for c in chunks)
    for key in ("content", "page", "heading", "chunk_index"):
        assert key in chunks[0]


def test_empty_pages_are_skipped() -> None:
    pages = _pages("", "   ", "Real content here " * 30)
    chunks = chunk_pages(pages)
    assert all(c["content"].strip() != "" for c in chunks)
    assert all(c["page"] == 3 for c in chunks)


def test_single_short_page_produces_one_chunk() -> None:
    pages = _pages("Short text.")
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Short text."


# ---------------------------------------------------------------------------
# Chunk size + overlap
# ---------------------------------------------------------------------------

def test_chunk_size_respected() -> None:
    text = "x" * 1000
    chunks = chunk_pages(_pages(text), chunk_size=200, overlap=0)
    for c in chunks:
        assert len(c["content"]) <= 200


def test_overlap_repeats_characters() -> None:
    text = "abcdefghij" * 20  # 200 chars
    chunks = chunk_pages(_pages(text), chunk_size=30, overlap=10)

    # Each chunk except the first should start with the last `overlap`
    # chars of the previous chunk's content
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1]["content"][-10:]
        curr_head = chunks[i]["content"][:10]
        assert prev_tail == curr_head, (
            f"Chunk {i} head {curr_head!r} != prev tail {prev_tail!r}"
        )


def test_no_overlap_produces_non_repeating_chunks() -> None:
    text = "a" * 100
    chunks = chunk_pages(_pages(text), chunk_size=25, overlap=0)
    # Expect exactly 4 chunks with no repeated content
    assert len(chunks) == 4
    assert all(len(c["content"]) == 25 for c in chunks)


def test_overlap_must_be_less_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages(_pages("some text"), chunk_size=100, overlap=100)


# ---------------------------------------------------------------------------
# Deterministic chunk_index ordering
# ---------------------------------------------------------------------------

def test_chunk_index_is_zero_based_and_sequential() -> None:
    text = "word " * 400
    chunks = chunk_pages(_pages(text), chunk_size=100, overlap=20)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_index_continues_across_pages() -> None:
    pages = _pages("a " * 300, "b " * 300)
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))

    # Ensure both pages contributed chunks
    page_nums = {c["page"] for c in chunks}
    assert page_nums == {1, 2}


def test_page_number_preserved_per_chunk() -> None:
    pages = _pages("Page one text " * 50, "Page two text " * 50)
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    page_1_chunks = [c for c in chunks if c["page"] == 1]
    page_2_chunks = [c for c in chunks if c["page"] == 2]
    assert len(page_1_chunks) > 0
    assert len(page_2_chunks) > 0


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

def test_heading_detected_title_case() -> None:
    text = "Introduction To Machine Learning\nThis section covers..."
    result = _best_heading(text)
    assert result == "Introduction To Machine Learning"


def test_heading_detected_all_caps() -> None:
    text = "METHODOLOGY\nWe used the following approach..."
    result = _best_heading(text)
    assert result == "METHODOLOGY"


def test_heading_detected_numbered() -> None:
    text = "1.2 Background\nThis chapter describes..."
    result = _best_heading(text)
    assert result == "1.2 Background"


def test_heading_not_detected_for_sentence() -> None:
    text = "This is a normal sentence that ends with a period.\nMore text."
    result = _best_heading(text)
    assert result is None


def test_heading_attached_to_chunk() -> None:
    text = "RESULTS\n" + ("data point analysis " * 60)
    chunks = chunk_pages(_pages(text), chunk_size=200, overlap=20)
    assert all(c["heading"] == "RESULTS" for c in chunks)


def test_no_heading_yields_none() -> None:
    text = "just some plain lowercase text with no heading at all. " * 20
    chunks = chunk_pages(_pages(text), chunk_size=200, overlap=20)
    assert all(c["heading"] is None for c in chunks)