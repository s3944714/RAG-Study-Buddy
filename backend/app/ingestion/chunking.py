import logging
import re
from typing import TypedDict

from app.ingestion.pdf_extractor import PageText

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heading detection heuristic
# ---------------------------------------------------------------------------
# A line is treated as a heading if it:
#   - Is 3–80 characters long
#   - Does not end with a sentence-terminating punctuation mark
#   - Is either Title Cased, ALL CAPS, or starts with a numbering pattern
#     like "1.", "1.1", "Chapter 2", "Section 3"
_NUMBERING_RE = re.compile(
    r"^(?:chapter|section|part|unit|module|appendix)?\s*\d+[\.\d]*\s+\w",
    re.IGNORECASE,
)


def _detect_heading(line: str) -> str | None:
    """Return the line if it looks like a heading, otherwise None."""
    line = line.strip()
    if not line or len(line) < 3 or len(line) > 80:
        return None
    if line[-1] in ".!?,;:":
        return None
    if (
        line.istitle()
        or line.isupper()
        or _NUMBERING_RE.match(line)
    ):
        return line
    return None


def _best_heading(text: str) -> str | None:
    """Scan the first few lines of a block and return the first detected heading."""
    for line in text.splitlines()[:6]:
        h = _detect_heading(line)
        if h:
            return h
    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class ChunkCandidate(TypedDict):
    content: str
    page: int
    heading: str | None
    chunk_index: int  # global 0-based index across all pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[ChunkCandidate]:
    """Split a list of PageText entries into overlapping ChunkCandidates.

    Args:
        pages:      Output of pdf_extractor.extract_text_from_pdf().
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters to repeat at the start of each
                    successive chunk (must be < chunk_size).

    Returns:
        Ordered list of ChunkCandidates with deterministic chunk_index values.

    Raises:
        ValueError: If overlap >= chunk_size.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    candidates: list[ChunkCandidate] = []
    global_index = 0

    for page in pages:
        text = page["text"].strip()
        page_num = page["page_number"]

        if not text:
            logger.debug("Page %d is empty — skipping", page_num)
            continue

        heading = _best_heading(text)
        step = chunk_size - overlap
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                candidates.append(
                    ChunkCandidate(
                        content=chunk_text,
                        page=page_num,
                        heading=heading,
                        chunk_index=global_index,
                    )
                )
                global_index += 1

            start += step

    logger.info(
        "Chunked %d pages → %d chunks (size=%d overlap=%d)",
        len(pages), len(candidates), chunk_size, overlap,
    )
    return candidates