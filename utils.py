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
div.stButton > button[kind="secondary"] {
    background-color: #0F172A !important;
    color: #F8FAFC !important;
    border: 1.5px solid #334155 !important;
}
div.stButton > button[kind="secondary"]:hover {
    border-color: #6366F1 !important;
    color: #FFFFFF !important;
}

/* Radio buttons in Quizzes */
div[data-testid="stRadio"] > div {
    background: #080B11 !important;
    padding: 12px 18px !important;
    border-radius: 14px !important;
    border: 1px solid #1E293B !important;
    margin-top: 8px !important;
}
div[data-testid="stRadio"] label span {
    color: #F8FAFC !important;
    font-size: 0.96rem !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    border-bottom: 2px solid #1E293B !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 700 !important;
    color: #94A3B8 !important;
}
.stTabs [aria-selected="true"] {
    color: #818CF8 !important;
    border-bottom: 3px solid #6366F1 !important;
}
</style>
"""


def apply_custom_styles():
    """Injects high-contrast fintech CSS into the Streamlit application."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state():
    """Ensures all state variables exist with safe defaults."""
    defaults: dict[str, Any] = {
        "page": "analyze",
        "subject": "Biology",
        "exam": "MDCAT",
        "query": "",
        "retrieved_chunks": [],
        "analysis": None,
        "quiz_questions": [],
        "quiz_answers": {},
        "quiz_submitted": False,
        "quiz_result": None,
        "quiz_start_time": None,
        "student_accuracy": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def navigate_to(page_name: str):
    """Sets the active navigation page and triggers rerun."""
    st.session_state.page = page_name
    st.rerun()


def render_fintech_priority_ring(score: int, label: str) -> str:
    """Generates an SVG circular progress ring styled for solid black."""
    score = max(0, min(100, score))
    radius = 54
    circumference = 2 * 3.1415926535 * radius
    stroke_dashoffset = circumference - (score / 100.0) * circumference

    if score >= 75:
        color = "#818CF8"   # High priority indigo
        bg_soft = "rgba(99, 102, 241, 0.15)"
    elif score >= 50:
        color = "#38BDF8"   # Medium priority sky blue
        bg_soft = "rgba(56, 189, 248, 0.15)"
    else:
        color = "#94A3B8"   # Slate
        bg_soft = "rgba(148, 163, 184, 0.15)"

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 12px 0;">
        <div style="position: relative; width: 136px; height: 136px; display: flex; align-items: center; justify-content: center;">
            <svg width="136" height="136" viewBox="0 0 136 136" style="transform: rotate(-90deg);">
                <circle cx="68" cy="68" r="{radius}" fill="none" stroke="#1E293B" stroke-width="10" />
                <circle cx="68" cy="68" r="{radius}" fill="none" stroke="{color}" stroke-width="10"
                    stroke-dasharray="{circumference:.2f}"
                    stroke-dashoffset="{stroke_dashoffset:.2f}"
                    stroke-linecap="round"
                    style="transition: stroke-dashoffset 0.8s ease;" />
            </svg>
            <div style="position: absolute; text-align: center; display: flex; flex-direction: column; align-items: center;">
                <span style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; line-height: 1;">{score}</span>
                <span style="font-size: 0.68rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 3px;">PRIORITY</span>
            </div>
        </div>
        <div style="margin-top: 10px; background: {bg_soft}; color: {color}; font-weight: 800; font-size: 0.78rem; padding: 4px 14px; border-radius: 9999px; letter-spacing: 0.06em; text-transform: uppercase; border: 1px solid {color}40;">
            {label}
        </div>
    </div>
    """


def render_fintech_score_ring(score: int, total: int = 10) -> str:
    """Generates an SVG circular progress ring for the 10-MCQ score."""
    percentage = int((score / total) * 100) if total > 0 else 0
    radius = 54
    circumference = 2 * 3.1415926535 * radius
    stroke_dashoffset = circumference - (percentage / 100.0) * circumference

    if percentage >= 80:
        color = "#4ADE80"   # High accuracy green
    elif percentage >= 50:
        color = "#38BDF8"   # Medium sky blue
    else:
        color = "#F87171"   # Low red

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 12px 0;">
        <div style="position: relative; width: 136px; height: 136px; display: flex; align-items: center; justify-content: center;">
            <svg width="136" height="136" viewBox="0 0 136 136" style="transform: rotate(-90deg);">
                <circle cx="68" cy="68" r="{radius}" fill="none" stroke="#1E293B" stroke-width="10" />
                <circle cx="68" cy="68" r="{radius}" fill="none" stroke="{color}" stroke-width="10"
                    stroke-dasharray="{circumference:.2f}"
                    stroke-dashoffset="{stroke_dashoffset:.2f}"
                    stroke-linecap="round"
                    style="transition: stroke-dashoffset 0.8s ease;" />
            </svg>
            <div style="position: absolute; text-align: center; display: flex; flex-direction: column; align-items: center;">
                <span style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; line-height: 1;">{score}<span style="font-size: 1.1rem; color: #94A3B8;">/{total}</span></span>
                <span style="font-size: 0.68rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 3px;">SCORE</span>
            </div>
        </div>
    </div>
    """


def render_concept_diagram_html(diagram_data: dict[str, Any]) -> str:
    """Renders an interactive concept flowchart with pure black and dark carbon boxes."""
    if not diagram_data or not isinstance(diagram_data, dict):
        return "<div class='fintech-card'>No diagram available.</div>"

    title = diagram_data.get("title", "Concept Flowchart")
    steps = diagram_data.get("steps", [])

    steps_html = ""
    for idx, step in enumerate(steps):
        step_title = step.get("step_title", f"Stage {idx + 1}")
        desc = step.get("description", "")
        callout = step.get("exam_callout", "")

        callout_badge = ""
        if callout:
            callout_badge = f"""
            <div style="margin-top: 8px; font-size: 0.76rem; font-weight: 700; color: #A5B4FC; background: rgba(99, 102, 241, 0.2); padding: 4px 10px; border-radius: 6px; display: inline-block; border: 1px solid rgba(99, 102, 241, 0.4);">
                ★ {callout}
            </div>
            """

        arrow_html = ""
        if idx < len(steps) - 1:
            arrow_html = """
            <div style="display: flex; justify-content: center; align-items: center; margin: 6px 0; color: #818CF8; font-size: 1.3rem; font-weight: 800;">
                ↓
            </div>
            """

        steps_html += f"""
        <div style="background: #080B11; border: 1.5px solid #1E293B; border-radius: 14px; padding: 16px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.5);">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span style="background: #6366F1; color: #FFFFFF; font-weight: 800; font-size: 0.75rem; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    {idx + 1}
                </span>
                <span style="font-weight: 800; font-size: 1.02rem; color: #FFFFFF;">
                    {step_title}
                </span>
            </div>
            <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.5; margin-left: 34px;">
                {desc}
                {callout_badge}
            </div>
        </div>
        {arrow_html}
        """

    return f"""
    <div class="fintech-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <h3 style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin: 0;">{title}</h3>
            <span class="badge-verified">VISUAL LOGIC FLOW</span>
        </div>
        <div style="display: flex; flex-direction: column;">
            {steps_html}
        </div>
    </div>
    """


def get_quiz_timer_state(total_seconds: int = 600) -> tuple[int, str, bool]:
    """Manages the 10-minute countdown timer state."""
    if st.session_state.quiz_start_time is None:
        st.session_state.quiz_start_time = time.time()

    elapsed = int(time.time() - st.session_state.quiz_start_time)
    remaining = max(0, total_seconds - elapsed)

    mins = remaining // 60
    secs = remaining % 60
    timer_str = f"{mins:02d}:{secs:02d}"
    is_expired = (remaining == 0)

    return remaining, timer_str, is_expired


def reset_quiz_timer():
    """Resets the quiz timer clock."""
    st.session_state.quiz_start_time = time.time()
