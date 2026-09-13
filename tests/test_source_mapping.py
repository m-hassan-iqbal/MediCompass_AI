"""
Unit tests for source ID validation, anti-hallucination sanitization, and deterministic fingerprinting.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_ingestion import DocumentChunk, compute_kb_fingerprint
from rag_engine import GroqClient


def test_sanitize_unknown_source_ids():
    groq = GroqClient()
    retrieved = [
        DocumentChunk(
            id="real_chunk_A",
            text="Text A",
            title="Book A",
            source_type="Punjab Book",
            subject="Biology",
            exam=["MDCAT"],
            chapter="",
            section="",
            page="12",
            year="N/A",
        ),
        DocumentChunk(
            id="real_chunk_B",
            text="Text B",
            title="Book B",
            source_type="Federal Book",
            subject="Biology",
            exam=["MDCAT"],
            chapter="",
            section="",
            page="34",
            year="N/A",
        ),
    ]

    raw_analysis = {
        "priority_score": 80,
        "source_ids": ["real_chunk_A", "hallucinated_chunk_xyz", "random_page_99", "real_chunk_B"],
    }

    sanitized = groq._validate_and_sanitize_analysis(raw_analysis, retrieved)
    assert "hallucinated_chunk_xyz" not in sanitized["source_ids"]
    assert "random_page_99" not in sanitized["source_ids"]
    assert sanitized["source_ids"] == ["real_chunk_A", "real_chunk_B"]


def test_fingerprint_deterministic(tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    f1 = sources_dir / "test.pdf"
    f1.write_bytes(b"%PDF-1.4 mock content")

    fp1 = compute_kb_fingerprint(str(sources_dir))
    fp2 = compute_kb_fingerprint(str(sources_dir))
    assert fp1 == fp2
    assert len(fp1) == 16
