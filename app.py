"""
MediCompass AI - Master Streamlit Application
Turning exam information into study intelligence for MDCAT and NUMS aspirants.
"""
import os
import time
import streamlit as st
from data_ingestion import (
    build_knowledge_base_chunks,
    compute_kb_fingerprint,
    sync_google_drive_public_folders,
)
from quiz_engine import (
    QuizGenerator,
    grade_quiz_submission,
)
from rag_engine import (
    GroqClient,
    LocalVectorStore,
)
from utils import (
    apply_custom_styles,
    get_quiz_timer_state,
    init_session_state,
    navigate_to,
    render_concept_diagram_html,
    render_fintech_priority_ring,
