from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.pdf_extractor import extract_text_from_pdf


# ---------------------------------------------------------------------------
# Minimal valid PDF builder (reused from test_upload pattern)
# ---------------------------------------------------------------------------

def _make_single_page_pdf(content: str = "Hello, world!") -> bytes:
    """Build the smallest PDF that pypdf can parse with real text content."""
    # We use reportlab-style raw PDF assembly — no extra deps needed.
    # The stream contains a BT/ET text block that pypdf can extract.
    stream = (
        f"BT\n/F1 12 Tf\n72 720 Td\n({content}) Tj\nET"
    ).encode()

    stream_obj = (
        b"4 0 obj\n"
        b"<< /Length " + str(len(stream)).encode() + b" >>\n"
        b"stream\n" + stream + b"\nendstream\nendobj\n"
    )

    header = b"%PDF-1.4\n"
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R\n"
        b"   /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R\n"
        b"   /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
        b"/BaseFont /Helvetica >> >> >> >>\n"
        b">>\nendobj\n"
    )

    body = header + obj1 + obj2 + obj3 + stream_obj

    xref_pos = len(body)
    xref = (
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        + _offset(body, obj1) +
        _offset(body, obj2) +
        _offset(body, obj3) +
        _offset(body, stream_obj)
    )
    trailer = (
        b"trailer\n<< /Size 5 /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return body + xref + trailer


def _offset(full: bytes, chunk: bytes) -> bytes:
    pos = full.find(chunk)
    return f"{pos:010d} 00000 n \n".encode()


def _make_empty_page_pdf() -> bytes:
    """PDF with one page that has an empty content stream."""
    header = b"%PDF-1.4\n"
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\n"
        b">>\nendobj\n"
    )
    body = header + obj1 + obj2 + obj3
    xref_pos = len(body)
    xref = (
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        + _offset(body, obj1)
        + _offset(body, obj2)
        + _offset(body, obj3)
    )
    trailer = (
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return body + xref + trailer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extract_returns_list_of_page_dicts() -> None:
    pdf_bytes = _make_single_page_pdf("Study notes page one")
    pages = extract_text_from_pdf(pdf_bytes)

    assert isinstance(pages, list)
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert isinstance(pages[0]["text"], str)


def test_page_numbers_are_one_based() -> None:
    pdf_bytes = _make_single_page_pdf("First page")
    pages = extract_text_from_pdf(pdf_bytes)
    assert pages[0]["page_number"] == 1


def test_empty_page_included_with_blank_text() -> None:
    """Pages with no extractable text must still appear in the result."""
    pdf_bytes = _make_empty_page_pdf()
    pages = extract_text_from_pdf(pdf_bytes)

    assert len(pages) == 1
    assert pages[0]["text"] == ""


def test_accepts_path_object(tmp_path: Path) -> None:
    pdf_bytes = _make_single_page_pdf("Path-based test")
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(pdf_bytes)

    pages = extract_text_from_pdf(pdf_file)
    assert len(pages) == 1


def test_accepts_string_path(tmp_path: Path) -> None:
    pdf_bytes = _make_single_page_pdf("String path test")
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(pdf_bytes)

    pages = extract_text_from_pdf(str(pdf_file))
    assert len(pages) == 1


def test_invalid_bytes_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Could not read PDF"):
        extract_text_from_pdf(b"this is not a pdf")


def test_extraction_via_mock() -> None:
    """Mock PdfReader to test multi-page logic without a real file."""
    mock_page_1 = MagicMock()
    mock_page_1.extract_text.return_value = "  Introduction  "
    mock_page_2 = MagicMock()
    mock_page_2.extract_text.return_value = ""   # empty page
    mock_page_3 = MagicMock()
    mock_page_3.extract_text.return_value = "Conclusion"

    mock_reader = MagicMock()
    mock_reader.pages = [mock_page_1, mock_page_2, mock_page_3]

    with patch("app.ingestion.pdf_extractor.PdfReader", return_value=mock_reader):
        pages = extract_text_from_pdf(b"%PDF-1.4 fake")

    assert len(pages) == 3
    assert pages[0] == {"page_number": 1, "text": "Introduction"}
    assert pages[1] == {"page_number": 2, "text": ""}
    assert pages[2] == {"page_number": 3, "text": "Conclusion"}