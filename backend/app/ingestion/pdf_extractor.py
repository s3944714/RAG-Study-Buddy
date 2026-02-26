import logging
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


class PageText(TypedDict):
    page_number: int   # 1-based
    text: str


def extract_text_from_pdf(source: str | Path | bytes) -> list[PageText]:
    """Extract text from a PDF, one entry per page.

    Args:
        source: A file path (str or Path) or raw PDF bytes.

    Returns:
        A list of PageText dicts, one per page.
        Pages with no extractable text are included with text="".

    Raises:
        ValueError: If the source cannot be read as a PDF.
    """
    try:
        if isinstance(source, bytes):
            import io
            reader = PdfReader(io.BytesIO(source))
        else:
            reader = PdfReader(str(source))
    except PdfReadError as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc

    pages: list[PageText] = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
            text = raw.strip()
        except Exception:
            logger.warning("Failed to extract text from page %d — skipping", i)
            text = ""

        if not text:
            logger.debug("Page %d is empty or has no extractable text", i)

        pages.append(PageText(page_number=i, text=text))

    logger.info("Extracted %d pages from PDF (%d non-empty)",
                len(pages), sum(1 for p in pages if p["text"]))
    return pages