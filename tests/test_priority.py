"""
Unit tests for Evidence-Based Study Priority formula, boundary conditions, and labeling.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_engine import GroqClient, DocumentChunk


def test_priority_score_bounds_and_labeling():
    groq = GroqClient()
    mock_chunks = [
        DocumentChunk(
            id="test_chunk_1",
            text="Mock text",
            title="Title",
            source_type="Punjab Book",
            subject="Biology",
            exam=["MDCAT"],
            chapter="Enzymes",
            section="",
            page="1",
            year="N/A",
        )
    ]

    analysis_overflow = {"priority_score": 150, "source_ids": ["test_chunk_1"]}
    sanitized = groq._validate_and_sanitize_analysis(analysis_overflow, mock_chunks)
    assert sanitized["priority_score"] == 100
    assert sanitized["priority_label"] == "STUDY NOW"

    analysis_underflow = {"priority_score": -20, "source_ids": ["test_chunk_1"]}
    sanitized_low = groq._validate_and_sanitize_analysis(analysis_underflow, mock_chunks)
    assert sanitized_low["priority_score"] == 0
    assert sanitized_low["priority_label"] == "LOWER PRIORITY"

    analysis_med = {"priority_score": 65, "source_ids": ["test_chunk_1"]}
    sanitized_med = groq._validate_and_sanitize_analysis(analysis_med, mock_chunks)
    assert sanitized_med["priority_score"] == 65
    assert sanitized_med["priority_label"] == "REVIEW SOON"


def test_demo_analysis_priority():
    groq = GroqClient()
    analysis = groq._generate_verified_demo_analysis("Enzymes", "Biology", "MDCAT", [])
    score = analysis["priority_score"]
    label = analysis["priority_label"]

    assert 0 <= score <= 100
    assert label in ["STUDY NOW", "REVIEW SOON", "LOWER PRIORITY"]
    assert "personal performance" in analysis["priority_reason"].lower()
