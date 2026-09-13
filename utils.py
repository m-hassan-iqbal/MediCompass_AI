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

/* Header & Navigation Bar - Semi-transparent dark blur so Streamlit Cloud controls remain visible */
#MainMenu {visibility: hidden;}
footer {display: none;}
header[data-testid="stHeader"] {
    background-color: rgba(5, 7, 14, 0.8) !important;
    backdrop-filter: blur(8px) !important;
}

/* Clear, High-Contrast Error, Warning & Alert banners */
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
    background-color: #1E293B !important;
    border-color: #6366F1 !important;
}
div.stButton > button[kind="secondary"] * {
    color: #F8FAFC !important;
}

/* Quiz Option Radio Cards */
div[data-testid="stRadio"] > div {
    gap: 12px;
}
div[data-testid="stRadio"] label {
    background-color: #0F172A !important;
    border: 1.5px solid #1E293B !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    cursor: pointer;
    transition: all 0.15s ease-in-out;
}
div[data-testid="stRadio"] label:hover {
    border-color: #6366F1 !important;
    background-color: #18182E !important;
}
div[data-testid="stRadio"] label * {
    color: #F8FAFC !important;
}

/* Metric KPI Styling */
.kpi-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: #0F172A;
    border: 1.5px solid #1E293B;
    border-radius: 16px;
    padding: 18px 22px;
}
.kpi-val {
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
}
.kpi-label {
    font-size: 0.82rem;
    color: #94A3B8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tabs Dark styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #080B11;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #1E293B;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94A3B8 !important;
    font-weight: 700 !important;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
}
</style>
"""


def apply_custom_styles():
    """Injects custom CSS into the active Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state():
    """Initializes persistent application session state defaults."""
    defaults = {
        "page": "analyze",
        "subject": "Biology",
        "exam": "MDCAT",
        "query": "",
        "analysis": None,
        "quiz": None,
        "quiz_submitted": False,
        "user_answers": {},
        "quiz_score": 0,
        "quiz_start_time": None,
        "quiz_time_taken": 0,
        "viewed_sources": [],
        "groq_api_key": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def navigate_to(page_name: str):
    """Updates the active navigation page in session state."""
    st.session_state.page = page_name
    st.rerun()


def render_fintech_priority_ring(priority_level: str, size: int = 120) -> str:
    """
    Renders an SVG ring indicator for exam priority (High/Medium/Low).
    """
    levels = {
        "high": {"percent": 92, "color": "#EF4444", "label": "HIGH", "bg": "#450A0A"},
        "medium": {"percent": 65, "color": "#F59E0B", "label": "MED", "bg": "#451A03"},
        "low": {"percent": 35, "color": "#10B981", "label": "LOW", "bg": "#022C22"},
    }
    cfg = levels.get(priority_level.lower(), levels["high"])
    stroke_dashoffset = 283 - (283 * cfg["percent"] / 100)

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <svg width="{size}" height="{size}" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" fill="none" stroke="#1E293B" stroke-width="8"/>
            <circle cx="50" cy="50" r="45" fill="none" stroke="{cfg['color']}" stroke-width="8"
                    stroke-dasharray="283" stroke-dashoffset="{stroke_dashoffset}"
                    stroke-linecap="round" transform="rotate(-90 50 50)"/>
            <text x="50" y="54" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif"
                  font-size="16" font-weight="800" fill="#FFFFFF">{cfg['label']}</text>
        </svg>
        <div style="font-size: 0.72rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; margin-top: 4px; letter-spacing: 0.05em;">
            EXAM YIELD
        </div>
    </div>
    """


def render_fintech_score_ring(score: int, total: int = 10, size: int = 130) -> str:
    """
    Renders an SVG score progress ring for the quiz results screen.
    """
    pct = int((score / total) * 100) if total > 0 else 0
    stroke_dashoffset = 283 - (283 * pct / 100)
    color = "#10B981" if pct >= 70 else ("#F59E0B" if pct >= 50 else "#EF4444")

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <svg width="{size}" height="{size}" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" fill="none" stroke="#1E293B" stroke-width="8"/>
            <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="8"
                    stroke-dasharray="283" stroke-dashoffset="{stroke_dashoffset}"
                    stroke-linecap="round" transform="rotate(-90 50 50)"/>
            <text x="50" y="48" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif"
                  font-size="20" font-weight="800" fill="#FFFFFF">{score}/{total}</text>
            <text x="50" y="65" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif"
                  font-size="10" font-weight="700" fill="#94A3B8">{pct}%</text>
        </svg>
        <div style="font-size: 0.75rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; margin-top: 6px; letter-spacing: 0.05em;">
            ACCURACY SCORE
        </div>
    </div>
    """


def render_concept_diagram_html(diagram_spec: dict[str, Any]) -> str:
    """
    Renders an HTML/SVG flowchart card for Concept Diagrams.
    """
    steps = diagram_spec.get("steps", [])
    if not steps:
        steps = [
            {"title": "Initial State", "desc": "Substrate approaches active site"},
            {"title": "Inhibitor Action", "desc": "Inhibitor binds or alters conformation"},
            {"title": "Kinetic Outcome", "desc": "Km or Vmax changes accordingly"},
        ]

    steps_html = ""
    for idx, step in enumerate(steps):
        is_last = (idx == len(steps) - 1)
        steps_html += f"""
        <div style="flex: 1; min-width: 170px; background-color: #1E293B; border: 1.5px solid #334155; border-radius: 14px; padding: 16px; position: relative;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="background-color: #6366F1; color: #FFFFFF; font-size: 0.72rem; font-weight: 800; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    {idx + 1}
                </span>
                <span style="font-size: 0.88rem; font-weight: 800; color: #FFFFFF;">
                    {step.get('title', 'Step')}
                </span>
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.4;">
                {step.get('desc', '')}
            </div>
        </div>
        """
        if not is_last:
            steps_html += """
            <div style="display: flex; align-items: center; justify-content: center; padding: 0 6px; color: #6366F1; font-size: 1.3rem; font-weight: 800;">
                →
            </div>
            """

    return f"""
    <div style="background-color: #0F172A; border: 1.5px solid #1E293B; border-radius: 20px; padding: 22px; margin: 16px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div style="font-size: 0.82rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                {diagram_spec.get('type', 'CONCEPT FLOWCHART').upper()}
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 600;">
                {diagram_spec.get('title', 'Mental Model')}
            </div>
        </div>
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 8px;">
            {steps_html}
        </div>
    </div>
    """


def get_quiz_timer_state() -> int:
    """Tracks elapsed time during quiz session."""
    if st.session_state.quiz_start_time is None:
        st.session_state.quiz_start_time = time.time()
        return 0
    return int(time.time() - st.session_state.quiz_start_time)


def reset_quiz_timer():
    """Resets quiz timer on new quiz start."""
    st.session_state.quiz_start_time = time.time()
