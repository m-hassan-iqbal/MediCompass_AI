"""
MediCompass AI - UI Components, Fintech Visualizations, Diagram Renderer & Timer Logic
Provides custom CSS, fintech-style circular SVG rings, responsive concept flowcharts,
and robust session-state helpers tailored for Solid Black / Dark Mode.
"""

import time
from typing import Any
import streamlit as st

CUSTOM_CSS = """
<style>
/* Modern Fintech Intelligence Theme - Solid Black / Dark OLED */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --text-color: #F8FAFC !important;
    --background-color: #000000 !important;
    --secondary-background-color: #0F172A !important;
    --primary-color: #6366F1 !important;
}

/* Force solid black background and crisp white text */
html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #000000 !important;
    color: #F8FAFC !important;
}

/* Base text color overrides */
[data-testid="stAppViewContainer"] *,
[data-testid="stSidebar"] *,
[data-testid="stMarkdownContainer"] *,
.stMarkdown, .stText, p, span, div, label, li {
    color: #F8FAFC;
}

/* Headings in vibrant white */
h1, h2, h3, h4, h5, h6, .hero-title {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

/* Header & Navigation Bar - Semi-transparent dark blur */
#MainMenu {visibility: hidden;}
footer {display: none;}
header[data-testid="stHeader"] {
    background-color: rgba(5, 7, 14, 0.8) !important;
    backdrop-filter: blur(8px) !important;
}

/* High-Contrast Error, Warning & Alert banners */
[data-testid="stAlert"], .stAlert, [data-testid="stException"] {
    background-color: #1E1B4B !important;
    border: 1.5px solid #818CF8 !important;
    border-radius: 14px !important;
    color: #FFFFFF !important;
    padding: 16px !important;
    margin: 12px 0 !important;
}
[data-testid="stAlert"] *, .stAlert *, [data-testid="stException"] * {
    color: #FFFFFF !important;
}

/* Container padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1140px;
}

/* Sidebar styling - OLED solid dark */
[data-testid="stSidebar"] {
    background-color: #080B11 !important;
    border-right: 1.5px solid #1E293B !important;
}
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #F8FAFC !important;
}

/* Section titles in sidebar */
.sidebar-section-title {
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    color: #94A3B8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 8px !important;
}

/* Fintech card styles - dark carbon cards */
.fintech-card {
    background: #0F172A !important;
    border: 1.5px solid #1E293B !important;
    border-radius: 20px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 4px 24px -2px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.2s ease-in-out;
}
.fintech-card:hover {
    box-shadow: 0 8px 32px -4px rgba(99, 102, 241, 0.15) !important;
    border-color: #334155 !important;
}
.fintech-card * {
    color: #F8FAFC;
}

/* Badges with high contrast in dark mode */
.badge-verified {
    background-color: #1E3A8A !important;
    color: #93C5FD !important;
    padding: 5px 14px !important;
    border-radius: 9999px !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em !important;
    border: 1.5px solid #3B82F6 !important;
    display: inline-flex !important;
    align-items: center !important;
}

.badge-inference {
    background-color: #3B0764 !important;
    color: #D8B4FE !important;
    padding: 5px 14px !important;
    border-radius: 9999px !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em !important;
    border: 1.5px solid #8B5CF6 !important;
    display: inline-flex !important;
    align-items: center !important;
}

.badge-exam {
    background-color: #1E293B !important;
    color: #60A5FA !important;
    padding: 5px 12px !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    border: 1px solid #3B82F6 !important;
    margin-right: 6px !important;
}

/* Hero Section */
.hero-tag {
    color: #818CF8 !important;
    font-size: 0.88rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    margin-bottom: 8px !important;
}

.hero-title {
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    line-height: 1.25 !important;
    margin-bottom: 12px !important;
}

.hero-sub {
    font-size: 1.08rem !important;
    color: #94A3B8 !important;
    line-height: 1.6 !important;
    margin-bottom: 24px !important;
    font-weight: 500 !important;
}

/* Excerpt quote block */
.source-excerpt {
    background-color: #090D16 !important;
    border-left: 4px solid #6366F1 !important;
    padding: 14px 18px !important;
    border-radius: 0 12px 12px 0 !important;
    font-size: 0.94rem !important;
    color: #CBD5E1 !important;
    font-style: italic !important;
    line-height: 1.6 !important;
    margin: 12px 0 !important;
    border-top: 1px solid #1E293B !important;
    border-right: 1px solid #1E293B !important;
    border-bottom: 1px solid #1E293B !important;
}

/* Memory hook highlight box */
.memory-box {
    background: #18182E !important;
    border: 1.5px solid #4338CA !important;
    border-radius: 16px !important;
    padding: 18px 22px !important;
    margin: 16px 0 !important;
}
.memory-box * {
    color: #E0E7FF !important;
}

/* Form Input & Widget Labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

/* Text Area and Selectbox Contrast in Solid Black */
.stTextArea textarea {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    border: 1.5px solid #334155 !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
.stTextArea textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3) !important;
}

div[data-baseweb="select"] {
    background-color: #0F172A !important;
    border: 1.5px solid #334155 !important;
    border-radius: 12px !important;
}
div[data-baseweb="select"] * {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
div[data-baseweb="popover"] ul {
    background-color: #0F172A !important;
}
div[data-baseweb="popover"] li {
    color: #FFFFFF !important;
}

/* Button contrast */
div.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
div.stButton > button[kind="primary"] {
    background-color: #6366F1 !important;
    color: #FFFFFF !important;
    border: none !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #4F46E5 !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
}
div.stButton > button[kind="primary"] * {
    color: #FFFFFF !important;
}
div.stButton > button[kind="secondary"] {
    background-color: #0F172A !important;
    color: #F8FAFC !important;
    border: 1.5px solid #334155 !important;
}
div.stButton > button[kind="secondary"]:hover {
    border-color: #6366F1 !important;
    background-color: #1E293B !important;
}

/* Radio button options */
div[role="radiogroup"] > label {
    background: #0F172A !important;
    border: 1.5px solid #1E293B !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
    cursor: pointer !important;
}
div[role="radiogroup"] > label * {
    color: #F8FAFC !important;
    font-weight: 600 !important;
}
div[role="radiogroup"] > label:hover {
    border-color: #6366F1 !important;
    background: #161F33 !important;
}

/* Tabs styling in solid black */
[data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid #1E293B !important;
}
[data-baseweb="tab"] {
    color: #94A3B8 !important;
    font-weight: 700 !important;
}
[aria-selected="true"] {
    color: #818CF8 !important;
}
</style>
"""


def apply_custom_styles():
    """Injects custom fintech solid black CSS into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_fintech_priority_ring(*args, **kwargs) -> str:
    """
    Renders an SVG priority ring gauge. Completely polymorphic to accept:
    (score, label), (label, score), or keywords: score=..., label=..., priority_level=...
    """
    score = 75
    label = "STUDY NOW"

    # 1. Process keyword arguments
    if "score" in kwargs and kwargs["score"] is not None:
        try:
            score = int(kwargs["score"])
        except (ValueError, TypeError):
            score = 75
    if "label" in kwargs and kwargs["label"] is not None:
        label = str(kwargs["label"])
    elif "priority_level" in kwargs and kwargs["priority_level"] is not None:
        label = str(kwargs["priority_level"])

    # 2. Process positional arguments
    if len(args) == 1:
        val = args[0]
        if isinstance(val, (int, float)):
            score = int(val)
        elif val is not None:
            label = str(val)
    elif len(args) >= 2:
        val1, val2 = args[0], args[1]
        if isinstance(val1, (int, float)):
            score = int(val1)
            label = str(val2 or "STUDY NOW")
        elif isinstance(val2, (int, float)):
            score = int(val2)
            label = str(val1 or "STUDY NOW")
        else:
            try:
                score = int(val1)
                label = str(val2 or "STUDY NOW")
            except (ValueError, TypeError):
                try:
                    score = int(val2)
                    label = str(val1 or "STUDY NOW")
                except (ValueError, TypeError):
                    score = 75
                    label = str(val1 or "STUDY NOW")

    # Safe score clamp
    score = max(0, min(100, score))
    label_clean = str(label).strip() or "STUDY NOW"
    label_lower = label_clean.lower()

    radius = 54
    circumference = 2 * 3.14159 * radius
    stroke_offset = circumference - (score / 100.0) * circumference

    if score >= 75 or "study" in label_lower or "high" in label_lower:
        color = "#EF4444"
        bg_ring = "#3B1219"
    elif score >= 50 or "medium" in label_lower or "moderate" in label_lower:
        color = "#F59E0B"
        bg_ring = "#3D2B0F"
    else:
        color = "#10B981"
        bg_ring = "#0F3323"

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px;">
        <div style="position: relative; width: 140px; height: 140px;">
            <svg width="140" height="140" viewBox="0 0 140 140" style="transform: rotate(-90deg);">
                <circle cx="70" cy="70" r="{radius}" fill="none" stroke="{bg_ring}" stroke-width="12" />
                <circle cx="70" cy="70" r="{radius}" fill="none" stroke="{color}" stroke-width="12"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{stroke_offset}" stroke-linecap="round" />
            </svg>
            <div style="position: absolute; top: 0; left: 0; width: 140px; height: 140px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <span style="font-size: 2rem; font-weight: 800; color: #FFFFFF; line-height: 1;">{score}</span>
                <span style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; margin-top: 2px;">/ 100</span>
            </div>
        </div>
        <div style="margin-top: 12px; text-align: center;">
            <span style="background-color: {bg_ring}; color: {color}; padding: 6px 14px; border-radius: 9999px; font-weight: 800; font-size: 0.8rem; letter-spacing: 0.05em; border: 1px solid {color}66;">
                {label_clean}
            </span>
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600; margin-top: 6px;">
                Evidence Study Priority
            </div>
        </div>
    </div>
    """


def render_fintech_score_ring(*args, **kwargs) -> str:
    """
    Renders score ring gauge. Supports (score, total, label), (score, label), or keyword args.
    """
    score = 0
    total = 10
    label = "Concept Score"

    if "score" in kwargs:
        try:
            score = int(kwargs["score"])
        except (ValueError, TypeError):
            pass
    if "total" in kwargs:
        try:
            total = int(kwargs["total"])
        except (ValueError, TypeError):
            pass
    if "label" in kwargs:
        label = str(kwargs["label"])

    if len(args) == 1:
        try:
            score = int(args[0])
        except (ValueError, TypeError):
            pass
    elif len(args) == 2:
        try:
            score = int(args[0])
        except (ValueError, TypeError):
            pass
        if isinstance(args[1], (int, float)):
            total = int(args[1])
        else:
            label = str(args[1])
    elif len(args) >= 3:
        try:
            score = int(args[0])
            total = int(args[1])
            label = str(args[2])
        except (ValueError, TypeError):
            pass

    pct = (score / float(total)) if total > 0 else 0
    radius = 54
    circumference = 2 * 3.14159 * radius
    stroke_offset = circumference - (pct * circumference)

    if score >= int(total * 0.8):
        color = "#10B981"
        bg_ring = "#0F3323"
    elif score >= int(total * 0.5):
        color = "#F59E0B"
        bg_ring = "#3D2B0F"
    else:
        color = "#EF4444"
        bg_ring = "#3B1219"

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px;">
        <div style="position: relative; width: 140px; height: 140px;">
            <svg width="140" height="140" viewBox="0 0 140 140" style="transform: rotate(-90deg);">
                <circle cx="70" cy="70" r="{radius}" fill="none" stroke="{bg_ring}" stroke-width="12" />
                <circle cx="70" cy="70" r="{radius}" fill="none" stroke="{color}" stroke-width="12"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{stroke_offset}" stroke-linecap="round" />
            </svg>
            <div style="position: absolute; top: 0; left: 0; width: 140px; height: 140px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <span style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; line-height: 1;">{score}</span>
                <span style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; margin-top: 2px;">/ {total}</span>
            </div>
        </div>
        <div style="margin-top: 12px; text-align: center;">
            <div style="font-size: 0.9rem; font-weight: 800; color: #FFFFFF;">
                {label}
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">
                {round(pct * 100)}% Accuracy
            </div>
        </div>
    </div>
    """


def render_concept_diagram_html(*args, **kwargs) -> str:
    """
    Renders visual concept flowchart HTML.
    Supports:
    - Single dict: render_concept_diagram_html(diagram_data)
    - Keyword arguments: render_concept_diagram_html(title=..., nodes=..., connections=..., memory_hook=...)
    - Positional arguments: render_concept_diagram_html(title, nodes, connections)
    """
    diagram_data: dict[str, Any] = {}
    if args:
        if isinstance(args[0], dict):
            diagram_data = dict(args[0])
        elif isinstance(args[0], str):
            diagram_data["title"] = args[0]
            if len(args) > 1 and isinstance(args[1], list):
                diagram_data["nodes"] = args[1]
            if len(args) > 2 and isinstance(args[2], list):
                diagram_data["connections"] = args[2]

    for k, v in kwargs.items():
        diagram_data[k] = v

    title = str(diagram_data.get("title") or "Visual Concept Flow")
    nodes = diagram_data.get("nodes") or []
    connections = diagram_data.get("connections") or []
    hook = str(diagram_data.get("memory_hook") or "")

    cards_html = []
    for i, node in enumerate(nodes):
        node_step = f"0{i+1}" if i < 9 else str(i+1)
        card_content = f"""
        <div style="flex: 1; min-width: 170px; max-width: 220px; background: #131B2E; border: 1.5px solid #334155; border-radius: 16px; padding: 16px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4); margin: 8px; text-align: center;">
            <div style="font-size: 0.7rem; font-weight: 800; color: #818CF8; letter-spacing: 0.1em; text-transform: uppercase;">STEP {node_step}</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF; margin-top: 6px; line-height: 1.4;">{node}</div>
        </div>
        """
        cards_html.append(card_content)

        if i < len(nodes) - 1:
            edge_label = ""
            if i < len(connections):
                edge_label = connections[i][2] if len(connections[i]) > 2 else ""

            arrow = f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 4px 0;">
                <span style="font-size: 0.65rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; background: #1E293B; padding: 2px 6px; border-radius: 4px; margin-bottom: 2px;">{edge_label}</span>
                <span style="color: #818CF8; font-size: 1.4rem; font-weight: 800;">→</span>
            </div>
            """
            cards_html.append(arrow)

    flow_content = "".join(cards_html)
    memory_rule_html = ""
    if hook:
        memory_rule_html = f"""
        <div style="margin-top: 24px; background: #18182E; border-radius: 12px; padding: 14px 20px; border-left: 4px solid #6366F1; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; text-transform: uppercase; letter-spacing: 0.05em;">CORE MEMORY RULE</span>
                <div style="font-size: 0.95rem; font-weight: 700; color: #E0E7FF; margin-top: 2px;">{hook}</div>
            </div>
            <span style="font-size: 1.5rem;">🧠</span>
        </div>
        """

    return f"""
    <div style="background: #0F172A; border: 1.5px solid #1E293B; border-radius: 20px; padding: 28px; box-shadow: 0 6px 24px -4px rgba(0, 0, 0, 0.4); margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E293B; padding-bottom: 14px; margin-bottom: 20px;">
            <div>
                <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">VISUAL CONCEPT FLOW</span>
                <h3 style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 0 0;">{title}</h3>
            </div>
            <span style="background-color: #1E1B4B; color: #C7D2FE; padding: 6px 14px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700; border: 1px solid #4338CA;">
                Infographic Architecture
            </span>
        </div>

        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 8px; padding: 10px 0;">
            {flow_content}
        </div>

        {memory_rule_html}
    </div>
    """


def get_quiz_timer_state(*args, **kwargs) -> tuple[int, str, bool]:
    """
    Returns (remaining_seconds, formatted_string, is_expired).
    Accepts total_seconds positionally, via kwargs total_seconds=..., or defaults to 600.
    """
    total_seconds = 600
    if args and isinstance(args[0], (int, float)):
        total_seconds = int(args[0])
    elif "total_seconds" in kwargs and kwargs["total_seconds"] is not None:
        try:
            total_seconds = int(kwargs["total_seconds"])
        except (ValueError, TypeError):
            total_seconds = 600

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
