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
/* Modern Fintech Intelligence Theme */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #172033;
}

.stApp {
    background-color: #F7F9FD;
}

/* Hide Streamlit default hamburger & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Container padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1140px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

/* Fintech card styles */
.fintech-card {
    background: #FFFFFF;
    border: 1px solid #EDF2F7;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px -2px rgba(23, 32, 51, 0.04);
    transition: all 0.2s ease-in-out;
}
.fintech-card:hover {
    box-shadow: 0 8px 30px -4px rgba(23, 32, 51, 0.08);
}

/* Pills & Badges */
.badge-verified {
    background-color: #EBF5FF;
    color: #2563EB;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    border: 1px solid #BFDBFE;
    display: inline-flex;
    align-items: center;
}

.badge-inference {
    background-color: #F5F3FF;
    color: #7C3AED;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    border: 1px solid #DDD6FE;
    display: inline-flex;
    align-items: center;
}

.badge-exam {
    background-color: #EFF6FF;
    color: #1D4ED8;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
}

.badge-priority-high {
    background-color: #FEF2F2;
    color: #DC2626;
    border: 1px solid #FCA5A5;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 800;
    font-size: 0.85rem;
}

.badge-priority-med {
    background-color: #FFFBEB;
    color: #D97706;
    border: 1px solid #FCD34D;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 800;
    font-size: 0.85rem;
}

/* Hero Section */
.hero-tag {
    color: #5368E9;
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 2.25rem;
    font-weight: 800;
    color: #172033;
    line-height: 1.25;
    margin-bottom: 12px;
}

.hero-sub {
    font-size: 1.05rem;
    color: #4A5568;
    line-height: 1.6;
    margin-bottom: 24px;
}

/* Excerpt quote block */
.source-excerpt {
    background-color: #F8FAFC;
    border-left: 4px solid #5368E9;
    padding: 14px 18px;
    border-radius: 0 12px 12px 0;
    font-size: 0.92rem;
    color: #334155;
    font-style: italic;
    line-height: 1.6;
    margin: 12px 0;
}

/* Memory hook highlight box */
.memory-box {
    background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
    border: 1px solid #C7D2FE;
    border-radius: 16px;
    padding: 18px 22px;
    margin: 16px 0;
}

/* Primary and secondary button improvements */
div.stButton > button:first-child {
    border-radius: 12px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s;
}

/* Radio button option cards */
div[role="radiogroup"] > label {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
}
div[role="radiogroup"] > label:hover {
    border-color: #5368E9;
    background: #F8FAFF;
}

</style>
"""


def apply_custom_styles():
    """Injects custom fintech CSS into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_fintech_priority_ring(score: int, label: str) -> str:
    """
    Renders an SVG circular gauge ring for Evidence-Based Study Priority.
    Score: 0 - 100.
    """
    radius = 54
    circumference = 2 * 3.14159 * radius
    stroke_offset = circumference - (score / 100.0) * circumference

    if score >= 75:
        color = "#DC2626"  # Red/Urgent Study Now
        bg_ring = "#FEE2E2"
    elif score >= 50:
        color = "#F59E0B"  # Amber Review Soon
        bg_ring = "#FEF3C7"
    else:
        color = "#10B981"  # Green Lower Priority
        bg_ring = "#D1FAE5"

    svg = f"""
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
    return svg


def render_fintech_score_ring(score: int, total: int = 10, label: str = "Concept Score") -> str:
    """
    Renders an SVG circular gauge ring for Quiz Results (e.g. 7 / 10).
    """
    pct = (score / float(total)) if total > 0 else 0
    radius = 54
    circumference = 2 * 3.14159 * radius
    stroke_offset = circumference - (pct * circumference)

    if score >= 8:
        color = "#10B981"  # Emerald Green
        bg_ring = "#D1FAE5"
    elif score >= 5:
        color = "#F59E0B"  # Amber
        bg_ring = "#FEF3C7"
    else:
        color = "#EF4444"  # Red
        bg_ring = "#FEE2E2"

    svg = f"""
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
    return svg


def render_concept_diagram_html(diagram_data: dict[str, Any]) -> str:
    """
    Renders a responsive, high-aesthetic educational flowchart/concept map
    with white rounded cards, soft blue/purple accents, and clean directional links.
    """
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

        # Insert connecting arrow between cards
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
    """
    Manages a 10-minute (600s) countdown timer using session state timestamps.
    Returns: (seconds_remaining, formatted_time_string, is_expired)
    """
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
    """Resets the quiz timer state."""
    st.session_state.quiz_started_at = time.time()


def init_session_state():
    """Initializes persistent application session state variables."""
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
    """Safely transitions between pages without losing analysis state."""
    st.session_state.page = page_name
    st.rerun()
