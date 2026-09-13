"""
Unit tests for text cleaning, chunking, and PDF page extraction.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_ingestion import clean_text, chunk_text, extract_pdf_pages


def test_clean_text():
    raw = "Hello   world!\r\n\r\n\r\nThis is   MDCAT.\n\n"
    cleaned = clean_text(raw)
    assert "  " not in cleaned
    assert "\r" not in cleaned
    assert "Hello world!" in cleaned
    assert "This is MDCAT." in cleaned


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short():
    short_text = "Enzymes are biological catalysts that speed up reactions."
    chunks = chunk_text(short_text, chunk_size=1200, overlap=200)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunk_text_overlap_and_size():
    paragraphs = [f"Paragraph {i}: " + ("word " * 50) + "\n\n" for i in range(10)]
    long_text = "".join(paragraphs)
    chunk_size = 400
    overlap = 100
    chunks = chunk_text(long_text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= chunk_size + 150

    if len(chunks) >= 2:
        overlap_found = False
        words_c0 = set(chunks[0].split()[-10:])
        words_c1 = set(chunks[1].split()[:15])
        if words_c0.intersection(words_c1):
            overlap_found = True
        assert overlap_found


def test_extract_pdf_pages_missing_file():
    pages = extract_pdf_pages("non_existent_file.pdf")
    assert pages == []
