"""
MediCompass AI - UI Components, Fintech Visualizations, Diagram Renderer & Timer Logic
Provides custom CSS, fintech-style circular SVG rings, responsive concept flowcharts,
and robust session-state helpers.
"""

import time
from typing import Any
import streamlit as st

CUSTOM_CSS = """
<style>
/* Modern Fintech Intelligence Theme - High Contrast Enforcement */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --text-color: #172033 !important;
    --background-color: #F7F9FD !important;
    --secondary-background-color: #FFFFFF !important;
    --primary-color: #5368E9 !important;
}

/* Force light background and dark text on the entire app container */
html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #F7F9FD !important;
    color: #172033 !important;
}

/* Override Streamlit Dark Mode text inheritance */
[data-testid="stAppViewContainer"] *,
[data-testid="stSidebar"] *,
[data-testid="stMarkdownContainer"] *,
.stMarkdown, .stText, p, span, div, label, li, h1, h2, h3, h4, h5, h6 {
    color: #172033;
}

/* Force headings to crisp dark navy */
h1, h2, h3, h4, h5, h6, .hero-title {
    color: #0F172A !important;
    font-weight: 800 !important;
}

/* Hide Streamlit default hamburger & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Container padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1140px;
}

/* Sidebar styling - crisp white with dark border */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1.5px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #172033 !important;
}

/* Section titles in sidebar */
.sidebar-section-title {
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    color: #334155 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 8px !important;
}

/* Fintech card styles */
.fintech-card {
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 20px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 4px 20px -2px rgba(23, 32, 51, 0.06) !important;
    transition: all 0.2s ease-in-out;
}
.fintech-card:hover {
    box-shadow: 0 8px 30px -4px rgba(23, 32, 51, 0.1) !important;
    border-color: #CBD5E1 !important;
}
.fintech-card * {
    color: #172033;
}

/* Pills & Badges with high contrast */
.badge-verified {
    background-color: #EFF6FF !important;
    color: #1D4ED8 !important;
    padding: 5px 14px !important;
    border-radius: 9999px !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em !important;
    border: 1.5px solid #93C5FD !important;
    display: inline-flex !important;
    align-items: center !important;
}

.badge-inference {
    background-color: #F5F3FF !important;
    color: #6D28D9 !important;
    padding: 5px 14px !important;
    border-radius: 9999px !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em !important;
    border: 1.5px solid #C4B5FD !important;
    display: inline-flex !important;
    align-items: center !important;
}

.badge-exam {
    background-color: #EEF2FF !important;
    color: #3730A3 !important;
    padding: 5px 12px !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    border: 1px solid #C7D2FE !important;
    margin-right: 6px !important;
}

/* Hero Section */
.hero-tag {
    color: #4338CA !important;
    font-size: 0.88rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    margin-bottom: 8px !important;
}

.hero-title {
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    line-height: 1.25 !important;
    margin-bottom: 12px !important;
}

.hero-sub {
    font-size: 1.08rem !important;
    color: #334155 !important;
    line-height: 1.6 !important;
    margin-bottom: 24px !important;
    font-weight: 500 !important;
}

/* Excerpt quote block */
.source-excerpt {
    background-color: #F8FAFC !important;
    border-left: 4px solid #5368E9 !important;
    padding: 14px 18px !important;
    border-radius: 0 12px 12px 0 !important;
    font-size: 0.94rem !important;
    color: #1E293B !important;
    font-style: italic !important;
    line-height: 1.6 !important;
    margin: 12px 0 !important;
    border-top: 1px solid #F1F5F9 !important;
    border-right: 1px solid #F1F5F9 !important;
    border-bottom: 1px solid #F1F5F9 !important;
}

/* Memory hook highlight box */
.memory-box {
    background: #EEF2FF !important;
    border: 1.5px solid #C7D2FE !important;
    border-radius: 16px !important;
    padding: 18px 22px !important;
    margin: 16px 0 !important;
}
.memory-box * {
    color: #1E1B4B !important;
}

/* Form Input & Widget Labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    color: #0F172A !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

/* Text Area and Selectbox Contrast */
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
.stTextArea textarea:focus {
    border-color: #5368E9 !important;
    box-shadow: 0 0 0 2px rgba(83, 104, 233, 0.2) !important;
}

div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 12px !important;
}
div[data-baseweb="select"] * {
    color: #0F172A !important;
    font-weight: 600 !important;
}

/* Button contrast */
div.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
div.stButton > button[kind="primary"] {
    background-color: #5368E9 !important;
    color: #FFFFFF !important;
    border: none !important;
}
div.stButton > button[kind="primary"] * {
    color: #FFFFFF !important;
}
div.stButton > button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1.5px solid #CBD5E1 !important;
}
div.stButton > button[kind="secondary"]:hover {
    border-color: #5368E9 !important;
    background-color: #F8FAFC !important;
}

/* Radio button options */
div[role="radiogroup"] > label {
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
    cursor: pointer !important;
}
div[role="radiogroup"] > label * {
    color: #1E293B !important;
    font-weight: 600 !important;
}
div[role="radiogroup"] > label:hover {
    border-color: #5368E9 !important;
    background: #F8FAFF !important;
}
</style>
"""


def apply_custom_styles():
    """Injects custom fintech CSS into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_fintech_priority_ring(score: int, label: str) -> str:
    radius = 54
    circumference = 2 * 3.14159 * radius
    stroke_offset = circumference - (score / 100.0) * circumference

    if score >= 75:
        color = "#DC2626"
        bg_ring = "#FEE2E2"
    elif score >= 50:
        color = "#F59E0B"
        bg_ring = "#FEF3C7"
    else:
        color = "#10B981"
        bg_ring = "#D1FAE5"

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px;">
        <div style="position: relative; width: 140px; height: 140px;">
            <svg width="140" height="140" viewBox="0 0 140 140" style="transform: rotate(-90deg);">
                <circle cx="70" cy="70" r="{radius}" fill="none" stroke="{bg_ring}" stroke-width="12" />
                <circle cx="70" cy="70" r="{radius}" fill="none" stroke="{color}" stroke-width="12"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{stroke_offset}" stroke-linecap="round" />
            </svg>
            <div style="position: absolute; top: 0; left: 0; width: 140px; height: 140px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <span style="font-size: 2rem; font-weight: 800; color: #172033; line-height: 1;">{score}</span>
                <span style="font-size: 0.75rem; font-weight: 700; color: #64748B; margin-top: 2px;">/ 100</span>
            </div>
        </div>
        <div style="margin-top: 12px; text-align: center;">
            <span style="background-color: {bg_ring}; color: {color}; padding: 6px 14px; border-radius: 9999px; font-weight: 800; font-size: 0.8rem; letter-spacing: 0.05em; border: 1px solid {color}33;">
                {label}
            </span>
            <div style="font-size: 0.75rem; color: #64748B; font-weight: 600; margin-top: 6px;">
                Evidence Study Priority
            </div>
        </div>
    </div>
    """


def render_fintech_score_ring(score: int, total: int = 10, label: str = "Concept Score") -> str:
    pct = (score / float(total)) if total > 0 else 0
    radius = 54
    circumference = 2 * 3.14159 * radius
    stroke_offset = circumference - (pct * circumference)

    if score >= 8:
        color = "#10B981"
        bg_ring = "#D1FAE5"
    elif score >= 5:
        color = "#F59E0B"
        bg_ring = "#FEF3C7"
    else:
        color = "#EF4444"
        bg_ring = "#FEE2E2"

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px;">
        <div style="position: relative; width: 140px; height: 140px;">
            <svg width="140" height="140" viewBox="0 0 140 140" style="transform: rotate(-90deg);">
                <circle cx="70" cy="70" r="{radius}" fill="none" stroke="{bg_ring}" stroke-width="12" />
                <circle cx="70" cy="70" r="{radius}" fill="none" stroke="{color}" stroke-width="12"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{stroke_offset}" stroke-linecap="round" />
            </svg>
            <div style="position: absolute; top: 0; left: 0; width: 140px; height: 140px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <span style="font-size: 2.1rem; font-weight: 800; color: #172033; line-height: 1;">{score}</span>
                <span style="font-size: 0.8rem; font-weight: 700; color: #64748B; margin-top: 2px;">/ {total}</span>
            </div>
        </div>
        <div style="margin-top: 12px; text-align: center;">
            <div style="font-size: 0.9rem; font-weight: 800; color: #172033;">
                {label}
            </div>
            <div style="font-size: 0.75rem; color: #64748B; font-weight: 600;">
                {round(pct * 100)}% Accuracy
            </div>
        </div>
    </div>
    """


def render_concept_diagram_html(diagram_data: dict[str, Any]) -> str:
    nodes = diagram_data.get("nodes", [])
    title = diagram_data.get("title", "Concept Map")
    hook = diagram_data.get("memory_hook", "")
    connections = diagram_data.get("connections", [])

    cards_html = []
    for i, node in enumerate(nodes):
        node_step = f"0{i+1}" if i < 9 else str(i+1)
        card_content = f"""
        <div style="flex: 1; min-width: 170px; max-width: 220px; background: #FFFFFF; border: 1.5px solid #E0E7FF; border-radius: 16px; padding: 16px; box-shadow: 0 4px 12px rgba(83, 104, 233, 0.06); margin: 8px; text-align: center;">
            <div style="font-size: 0.7rem; font-weight: 800; color: #5368E9; letter-spacing: 0.1em; text-transform: uppercase;">STEP {node_step}</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #172033; margin-top: 6px; line-height: 1.4;">{node}</div>
        </div>
        """
        cards_html.append(card_content)

        if i < len(nodes) - 1:
            edge_label = ""
            if i < len(connections):
                edge_label = connections[i][2] if len(connections[i]) > 2 else ""

            arrow = f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 4px 0;">
                <span style="font-size: 0.65rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; background: #EEF2FF; padding: 2px 6px; border-radius: 4px; margin-bottom: 2px;">{edge_label}</span>
                <span style="color: #5368E9; font-size: 1.4rem; font-weight: 800;">→</span>
            </div>
            """
            cards_html.append(arrow)

    flow_content = "".join(cards_html)

    return f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 20px; padding: 28px; box-shadow: 0 6px 24px -4px rgba(23, 32, 51, 0.05); margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F5F9; padding-bottom: 14px; margin-bottom: 20px;">
            <div>
                <span style="font-size: 0.75rem; font-weight: 800; color: #5368E9; letter-spacing: 0.08em; text-transform: uppercase;">VISUAL CONCEPT FLOW</span>
                <h3 style="font-size: 1.35rem; font-weight: 800; color: #172033; margin: 4px 0 0 0;">{title}</h3>
            </div>
            <span style="background-color: #EEF2FF; color: #4338CA; padding: 6px 14px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700; border: 1px solid #C7D2FE;">
                Infographic Architecture
            </span>
        </div>

        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 8px; padding: 10px 0;">
            {flow_content}
        </div>

        <div style="margin-top: 24px; background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%); border-radius: 12px; padding: 14px 20px; border-left: 4px solid #5368E9; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-size: 0.75rem; font-weight: 800; color: #5368E9; text-transform: uppercase; letter-spacing: 0.05em;">CORE MEMORY RULE</span>
                <div style="font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-top: 2px;">{hook}</div>
            </div>
            <span style="font-size: 1.5rem;">🧠</span>
        </div>
    </div>
    """


def get_quiz_timer_state(total_seconds: int = 600) -> tuple[int, str, bool]:
    if "quiz_started_at" not in st.session_state or st.session_state.quiz_started_at is None:
        st.session_state.quiz_started_at = time.time()

    elapsed = time.time() - st.session_state.quiz_started_at
    remaining = max(0, int(total_seconds - elapsed))

    mins = remaining // 60
    secs = remaining % 60
    formatted = f"{mins:02d}:{secs:02d}"
    is_expired = (remaining <= 0)

    return remaining, formatted, is_expired


def reset_quiz_timer():
    st.session_state.quiz_started_at = time.time()


def init_session_state():
    defaults = {
        "page": "analyze",
        "subject": "Biology",
        "exam": "MDCAT",
        "query": "",
        "retrieved_chunks": [],
        "analysis": None,
        "quiz_questions": [],
        "quiz_answers": {},
        "quiz_started_at": None,
        "quiz_submitted": False,
        "quiz_result": None,
        "student_accuracy": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def navigate_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()
