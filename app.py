"""
MediCompass AI - Master Streamlit Application
Turning exam information into study intelligence for MDCAT and NUMS aspirants.
"""

import os
import time
import logging
import streamlit as st

logger = logging.getLogger(__name__)

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
    render_fintech_score_ring,
    reset_quiz_timer,
)

# Set page layout and config
st.set_page_config(
    page_title="MediCompass AI - Exam Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply fintech CSS styling
apply_custom_styles()

# Initialize session state variables
init_session_state()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCES_DIR = os.path.join(BASE_DIR, "data", "sources")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "source_manifest.json")
PAST_PAPERS_PATH = os.path.join(BASE_DIR, "data", "past_papers.json")
CACHE_DIR = os.path.join(BASE_DIR, "cache")


# ─── KNOWLEDGE BASE CACHING & INITIALIZATION ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_initialized_vector_store():
    """Initializes and caches the LocalVectorStore based on deterministic fingerprint."""
    fp = compute_kb_fingerprint(SOURCES_DIR, MANIFEST_PATH)
    vs = LocalVectorStore(cache_dir=CACHE_DIR)

    # Try loading cached index first, self-heal if missing, empty, or method missing
    loaded = False
    if hasattr(vs, "load_cached_index"):
        try:
            loaded = vs.load_cached_index(fp)
        except Exception:
            loaded = False

    chunks_list = getattr(vs, "chunks", [])
    if not loaded or len(chunks_list) == 0:
        chunks = build_knowledge_base_chunks(
            sources_dir=SOURCES_DIR,
            manifest_path=MANIFEST_PATH,
            past_papers_path=PAST_PAPERS_PATH,
        )
        if hasattr(vs, "build_index"):
            vs.build_index(chunks, fp)
        else:
            vs.chunks = chunks

    return vs, fp


# ─── SECRETS & BACKEND CONFIGURATION ──────────────────────────────────────────
DEFAULT_DRIVE_URL = "https://drive.google.com/drive/folders/12JMNPPtw9ranu47PpEOFTqupACOHO66q?usp=sharing"


def get_configured_groq_key() -> str:
    """Dynamically detects Groq API key from Streamlit secrets (any case/section) or environment."""
    try:
        if hasattr(st, "secrets"):
            for k in ["GROQ_API_KEY", "groq_api_key", "GROQ_KEY", "groq_key"]:
                if k in st.secrets and st.secrets[k]:
                    val = str(st.secrets[k]).strip()
                    if val and not val.startswith("gsk_your_groq"):
                        return val
            for sec in ["groq", "general", "api_keys", "secrets"]:
                if sec in st.secrets and isinstance(st.secrets[sec], dict):
                    for k in ["GROQ_API_KEY", "groq_api_key", "api_key", "key"]:
                        if k in st.secrets[sec] and st.secrets[sec][k]:
                            val = str(st.secrets[sec][k]).strip()
                            if val and not val.startswith("gsk_your_groq"):
                                return val
    except Exception:
        pass
    env_val = os.environ.get("GROQ_API_KEY", "").strip()
    if env_val and not env_val.startswith("gsk_your_groq"):
        return env_val
    return ""


# Retrieve vector store immediately on startup (loads pre-built knowledge base)
vector_store, current_fingerprint = get_initialized_vector_store()

# Initialize Groq Client & Quiz Generator dynamically on each rerun
current_groq_key = get_configured_groq_key()
st.session_state.groq_api_key = current_groq_key

groq_model = "llama-3.3-70b-versatile"
try:
    if hasattr(st, "secrets") and "GROQ_MODEL" in st.secrets and st.secrets["GROQ_MODEL"]:
        groq_model = str(st.secrets["GROQ_MODEL"]).strip()
except Exception:
    pass
groq_model = os.environ.get("GROQ_MODEL", groq_model)
if groq_model in ["llama-3.1-8b-instant", "llama3.1-8b", "llama-3.1-8b"]:
    groq_model = "llama-3.3-70b-versatile"

groq_client = GroqClient(api_key=current_groq_key, model=groq_model)
quiz_gen = QuizGenerator(api_key=current_groq_key, model=groq_model)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 18px 0; border-bottom: 1.5px solid #1E293B; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem; color: #818CF8;">✦</span>
                <div>
                    <h2 style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 0; line-height: 1.1;">MediCompass</h2>
                    <span style="font-size: 0.72rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">AI • Exam Intelligence</span>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 8px; line-height: 1.4; font-weight: 500;">
                Turning exam information into study intelligence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation buttons
    st.markdown("<div style='font-size: 0.78rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;'>NAVIGATION</div>", unsafe_allow_html=True)

    pages = [
        ("⌂ Analyze", "analyze"),
        ("◈ Concept Intelligence", "concept"),
        ("🧠 Quick Diagram", "diagram"),
        ("✓ 10-MCQ Quiz", "quiz"),
        ("▤ Book Evidence", "evidence"),
    ]

    for label, page_key in pages:
        is_active = (st.session_state.page == page_key)
        btn_type = "primary" if is_active else "secondary"
        disabled = False
        if page_key != "analyze" and not st.session_state.analysis:
            disabled = True

        if st.button(label, key=f"nav_{page_key}", type=btn_type, disabled=disabled, use_container_width=True):
            navigate_to(page_key)

    st.markdown("---")

    # Current Target
    st.markdown("<div style='font-size: 0.78rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;'>CURRENT TARGET</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background: #0F172A; border: 1.5px solid #1E293B; border-radius: 12px; padding: 12px 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 0.82rem; color: #94A3B8; font-weight: 700;">Subject</span>
                <span style="font-size: 0.85rem; color: #F8FAFC; font-weight: 800;">{st.session_state.subject}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 0.82rem; color: #94A3B8; font-weight: 700;">Target Exam</span>
                <span style="font-size: 0.85rem; color: #818CF8; font-weight: 800;">{st.session_state.exam}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # System Status
    chunk_count = len(getattr(vector_store, "chunks", []))
    groq_ready = groq_client.is_configured()

    st.markdown("<div class='sidebar-section-title'>SYSTEM STATUS</div>", unsafe_allow_html=True)

    if chunk_count >= 1000:
        kb_text = f"<b style='color: #10B981;'>{chunk_count:,} Google Drive chunks</b>"
    else:
        kb_text = f"<b style='color: #F59E0B;'>{chunk_count} Multi-Subject chunks</b>"

    if groq_ready:
        ai_text = f"<span style='color: #4ADE80; font-weight: 800;'>🟢 Active Groq AI</span>"
    else:
        ai_text = "<span style='color: #38BDF8; font-weight: 800;'>⚡ Grounded Textbook RAG</span>"

    st.markdown(
        f"""
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.8; background: #0F172A; border: 1.5px solid #1E293B; border-radius: 12px; padding: 14px 16px;">
            <div>• Knowledge Base: {kb_text}</div>
            <div>• Official Books: <b style="color: #FFFFFF;">Punjab & Federal SNC</b></div>
            <div>• Past Papers: <b style="color: #FFFFFF;">MDCAT & NUMS 2021-25</b></div>
            <div>• RAG Engine: {ai_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── PAGE 1: ANALYZE / LANDING ───────────────────────────────────────────────
if st.session_state.page == "analyze":
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div class="hero-tag">PMDC CURRICULUM • MULTI-BOARD RAG INTELLIGENCE</div>
            <h1 class="hero-title">Transform Exam Complexity into Clinical Clarity</h1>
            <p class="hero-sub">
                Enter any MDCAT/NUMS past paper question or topic. MediCompass retrieves exact curriculum paragraphs
                from Punjab & Federal National textbooks, checks syllabus alignment, and synthesizes study intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        selected_subject = st.selectbox(
            "Select Subject",
            ["Biology", "Chemistry", "Physics"],
            index=["Biology", "Chemistry", "Physics"].index(st.session_state.subject)
            if st.session_state.subject in ["Biology", "Chemistry", "Physics"]
            else 0,
        )
        st.session_state.subject = selected_subject

    with col2:
        selected_exam = st.selectbox(
            "Target Examination",
            ["MDCAT", "NUMS", "MDCAT & NUMS"],
            index=["MDCAT", "NUMS", "MDCAT & NUMS"].index(st.session_state.exam)
            if st.session_state.exam in ["MDCAT", "NUMS", "MDCAT & NUMS"]
            else 0,
        )
        st.session_state.exam = selected_exam

    user_query = st.text_area(
        "Enter Past Paper Question or Concept Topic",
        value=st.session_state.query,
        height=130,
        placeholder="e.g. Mitochondria, Enzyme Inhibition (Competitive vs Non-competitive), or Newton's 3rd Law of Motion...",
    )

    c_btn1, c_btn2, c_btn3 = st.columns([1.5, 1, 1])
    with c_btn1:
        analyze_clicked = st.button("✦ Generate Concept Intelligence", type="primary", use_container_width=True)
    with c_btn2:
        if st.button("Try: Mitochondria", use_container_width=True):
            user_query = "Mitochondria cristae ATP matrix and Krebs cycle"
            st.session_state.subject = "Biology"
            analyze_clicked = True
    with c_btn3:
        if st.button("Try: Newton's 3rd Law", use_container_width=True):
            user_query = "Why action and reaction forces never cancel each other"
            st.session_state.subject = "Physics"
            analyze_clicked = True

    if analyze_clicked:
        if not user_query.strip():
            user_query = "Mitochondria structure and cellular respiration"

        st.session_state.query = user_query.strip()

        # Multi-stage progress indicator
        progress_box = st.empty()
        with progress_box.container():
            st.markdown(
                """
                <div class="fintech-card" style="text-align: center; padding: 24px;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #818CF8; margin-bottom: 8px;">
                        ✦ Retrieving Verified Book Evidence & Synthesizing...
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8;">
                        Searching Punjab Board, Federal Board, and official PMDC Past Papers...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # RAG Retrieval
            retrieved = vector_store.search(
                query=st.session_state.query,
                subject=st.session_state.subject,
                exam=st.session_state.exam,
                top_k=5,
            )
            st.session_state.retrieved_chunks = retrieved

            # Groq / LLM Analysis
            analysis = groq_client.analyze_concept(
                query=st.session_state.query,
                subject=st.session_state.subject,
                exam=st.session_state.exam,
                retrieved_chunks=retrieved,
            )
            st.session_state.analysis = analysis

            # Pre-generate 10-MCQ quiz questions
            quiz_questions = quiz_gen.generate_10_mcq_quiz(
                concept_title=analysis.get("concept_title", st.session_state.query),
                subject=st.session_state.subject,
                exam=st.session_state.exam,
                retrieved_chunks=retrieved,
            )
            st.session_state.quiz_questions = quiz_questions
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_result = None
            reset_quiz_timer()

        progress_box.empty()
        navigate_to("concept")


# ─── PAGE 2: CONCEPT INTELLIGENCE ─────────────────────────────────────────────
elif st.session_state.page == "concept":
    analysis = st.session_state.analysis
    if not analysis:
        st.info("No active analysis found. Please enter a question or concept to analyze.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    concept_title = analysis.get("concept_title", "Concept Analysis")

    # Breadcrumb / Navigation bar
    col_back, col_actions = st.columns([1, 2])
    with col_back:
        if st.button("← New Analysis", use_container_width=True):
            navigate_to("analyze")
    with col_actions:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🧠 Quick Memorize Diagram", use_container_width=True):
                navigate_to("diagram")
        with b2:
            if st.button("🎯 Attempt 10 Focused MCQs", type="primary", use_container_width=True):
                navigate_to("quiz")

    st.markdown(
        f"""
        <div style="margin: 18px 0 24px 0;">
            <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px;">
                <span class="badge-exam">{st.session_state.subject}</span>
                <span class="badge-exam">{st.session_state.exam}</span>
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin: 0;">
                {concept_title}
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Concept & Priority Row
    col_core, col_priority = st.columns([2.1, 1])

    with col_core:
        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                        CORE CONCEPT
                    </span>
                    <span class="badge-verified">VERIFIED</span>
                </div>
                <p style="font-size: 1.05rem; color: #F8FAFC; line-height: 1.65; margin: 0 0 16px 0; font-weight: 500;">
                    {analysis.get('core_explanation', '')}
                </p>
                <div class="memory-box">
                    <div style="font-size: 0.75rem; font-weight: 800; color: #818CF8; text-transform: uppercase; letter-spacing: 0.05em;">
                        REMEMBER IT FAST
                    </div>
                    <div style="font-size: 1.02rem; font-weight: 800; color: #FFFFFF; margin-top: 4px;">
                        {analysis.get('memory_hook', '')}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_priority:
        p_score = analysis.get("priority_score", 85)
        p_label = analysis.get("priority_label", "STUDY NOW")
        ring_svg = render_fintech_priority_ring(p_score, p_label)

        accuracy = st.session_state.student_accuracy
        if accuracy is None:
            perf_text = "⚠ Personal performance: Not recorded (take 10-MCQ quiz to calibrate)"
        else:
            perf_text = f"✓ Personal performance: {accuracy}% Accuracy recorded"

        st.markdown(
            f"""
            <div class="fintech-card" style="text-align: center;">
                {ring_svg}
                <div style="font-size: 0.8rem; color: #94A3B8; text-align: left; line-height: 1.5; margin-top: 14px; border-top: 1px solid #1E293B; padding-top: 12px;">
                    <div style="font-weight: 700; color: #CBD5E1; margin-bottom: 4px;">Priority Formula:</div>
                    <div>✓ PMDC Syllabus Weight: 40%</div>
                    <div>✓ Historical Past Paper Signal: 35%</div>
                    <div>{perf_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Step-by-Step Explanation
    deep_exp = analysis.get("deep_explanation", [])
    if deep_exp:
        st.markdown(
            """
            <div class="fintech-card">
                <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                    DEEP CONCEPTUAL DECONSTRUCTION
                </span>
                <h3 style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 6px 0 16px 0;">
                    How It Works & High-Yield Exam Mechanics
                </h3>
            """,
            unsafe_allow_html=True,
        )
        for pt in deep_exp:
            pt_clean = pt.replace("[VERIFIED]", "").replace("[INFERENCE]", "").strip()
            is_trap = "Trap" in pt or "Common" in pt
            icon = "⚠" if is_trap else "✦"
            border_color = "#F59E0B" if is_trap else "#334155"
            bg_color = "#18140B" if is_trap else "#090D16"

            st.markdown(
                f"""
                <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; display: flex; gap: 14px; align-items: flex-start;">
                    <span style="font-size: 1.1rem; color: {'#F59E0B' if is_trap else '#818CF8'}; font-weight: 800; line-height: 1.4;">{icon}</span>
                    <div style="font-size: 0.95rem; color: #E2E8F0; line-height: 1.6; font-weight: 500;">
                        {pt_clean}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Cross-Board Comparison: Punjab vs Federal
    st.markdown(
        """
        <div class="fintech-card">
            <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                CROSS-BOARD SYNTHESIS
            </span>
            <h3 style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 16px 0;">
                Curriculum Concordance: Punjab vs. Federal
            </h3>
        """,
        unsafe_allow_html=True,
    )

    c_pj, c_fed = st.columns(2)
    with c_pj:
        st.markdown(
            f"""
            <div style="background: #080B11; border: 1.5px solid #1E293B; border-radius: 14px; padding: 18px; height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    <span style="background: #064E3B; color: #A7F3D0; font-size: 0.72rem; font-weight: 800; padding: 3px 10px; border-radius: 6px;">PUNJAB TEXTBOOK</span>
                    <span style="font-size: 0.8rem; color: #94A3B8;">PTB Official</span>
                </div>
                <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
                    {analysis.get('punjab_synthesis', 'Grounded in Punjab Textbook Curriculum guidelines.')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_fed:
        st.markdown(
            f"""
            <div style="background: #080B11; border: 1.5px solid #1E293B; border-radius: 14px; padding: 18px; height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    <span style="background: #1E3A8A; color: #BFDBFE; font-size: 0.72rem; font-weight: 800; padding: 3px 10px; border-radius: 6px;">FEDERAL (NBF)</span>
                    <span style="font-size: 0.8rem; color: #94A3B8;">National Book Foundation</span>
                </div>
                <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
                    {analysis.get('federal_synthesis', 'Grounded in Federal Board National Curriculum standards.')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="margin-top: 16px; padding: 14px 18px; background: #131B2E; border-radius: 12px; border: 1px solid #334155; font-size: 0.9rem; color: #CBD5E1; line-height: 1.5;">
            <b style="color: #818CF8;">Key Synthesis Takeaway:</b> {analysis.get('synthesis_takeaway', '')}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Past Paper Evidence & Syllabus Check
    c_pp, c_syl = st.columns([1.6, 1])
    with c_pp:
        past_papers_ev = analysis.get("past_paper_evidence", [])
        pp_html = ""
        for pp in past_papers_ev:
            yr = pp.get("year", "Historical")
            ex = pp.get("exam", "MDCAT")
            sum_text = pp.get("summary", "")
            pp_html += f"""
            <div style="background: #080B11; border-left: 3px solid #6366F1; border-radius: 0 8px 8px 0; padding: 10px 14px; margin-bottom: 10px;">
                <div style="font-size: 0.78rem; font-weight: 800; color: #818CF8;">{ex} {yr} PAST PAPER</div>
                <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 4px; line-height: 1.4;">{sum_text}</div>
            </div>
            """

        st.markdown(
            f"""
            <div class="fintech-card">
                <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                    HISTORICAL EXAM EVIDENCE
                </span>
                <h3 style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 14px 0;">
                    Verified Past Paper Occurrences
                </h3>
                {pp_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_syl:
        syl_status = analysis.get("syllabus_status", "Covered")
        syl_details = analysis.get("syllabus_details", "Directly outlined in official PMDC curriculum specifications.")
        st.markdown(
            f"""
            <div class="fintech-card">
                <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                    SYLLABUS ALIGNMENT
                </span>
                <div style="margin: 12px 0;">
                    <span class="badge-verified" style="font-size: 0.9rem; padding: 6px 14px;">
                        ✓ {syl_status}
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5;">
                    {syl_details}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── PAGE 3: QUICK DIAGRAM ───────────────────────────────────────────────────
elif st.session_state.page == "diagram":
    analysis = st.session_state.analysis
    if not analysis or "diagram" not in analysis:
        st.info("No active diagram available. Please run an analysis first.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="hero-tag">COGNITIVE RETENTION • INFOGRAPHIC</div>
            <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">Quick Memorize Diagram</h1>
            <p style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">
                Visualizing cause-and-effect relationships allows instant mental recall during high-pressure medical entrance exams.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render interactive concept flowchart
    diagram_html = render_concept_diagram_html(analysis["diagram"])
    st.markdown(diagram_html, unsafe_allow_html=True)

    # Quick Recall Points
    quick_recall = analysis.get("quick_recall", [])
    if quick_recall:
        recall_cards = ""
        for pt in quick_recall:
            recall_cards += f"""
            <div style="background: #080B11; border: 1px solid #1E293B; border-radius: 14px; padding: 14px 18px; margin-bottom: 10px; display: flex; align-items: center; gap: 12px;">
                <span style="color: #818CF8; font-size: 1.2rem; font-weight: 800;">✓</span>
                <span style="font-size: 0.95rem; color: #FFFFFF; font-weight: 600;">{pt}</span>
            </div>
            """

        st.markdown(
            f"""
            <div class="fintech-card">
                <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                    RAPID EXAM RETRIEVAL CUES
                </span>
                <h3 style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 16px 0;">
                    Key Recall Anchors
                </h3>
                {recall_cards}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Actions
    c_diag1, c_diag2 = st.columns(2)
    with c_diag1:
        if st.button("← Return to Concept Intelligence", use_container_width=True):
            navigate_to("concept")
    with c_diag2:
        if st.button("🎯 Test Concept with 10 MCQs →", type="primary", use_container_width=True):
            navigate_to("quiz")


# ─── PAGE 4: 10-MCQ QUIZ ──────────────────────────────────────────────────────
elif st.session_state.page == "quiz":
    questions = st.session_state.quiz_questions
    if not questions:
        st.info("No active quiz generated. Please run an analysis first.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    analysis = st.session_state.analysis or {}
    concept_title = analysis.get("concept_title", st.session_state.query or "Concept Check")

    # Header and Live Countdown Timer
    c_head, c_timer = st.columns([2.2, 1])
    with c_head:
        st.markdown(
            f"""
            <div>
                <div class="hero-tag">10-MINUTE VERIFICATION CHECK</div>
                <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">10-MCQ Mastery Assessment</h1>
                <p style="font-size: 0.92rem; color: #94A3B8; margin-top: 4px;">
                    Topic: <b style="color: #FFFFFF;">{concept_title}</b> ({st.session_state.subject})
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_timer:
        remaining_sec, timer_str, is_expired = get_quiz_timer_state(total_seconds=600)
        timer_color = "#EF4444" if remaining_sec < 120 else "#818CF8"
        st.markdown(
            f"""
            <div class="fintech-card" style="padding: 14px 20px; text-align: center; margin-bottom: 0;">
                <div style="font-size: 0.72rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em;">TIME REMAINING</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {timer_color}; font-family: monospace; line-height: 1.1; margin-top: 4px;">{timer_str}</div>
                <div style="font-size: 0.72rem; color: #64748B;">Strict Exam Simulation (10 Mins)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Post-submission Results Card
    if st.session_state.quiz_submitted and st.session_state.quiz_result:
        res = st.session_state.quiz_result
        score = res["score"]
        total = res["total"]
        pct = res["percentage"]
        band = res["mastery_band"]

        st.markdown("---")
        c_score_ring, c_score_desc = st.columns([1, 2])
        with c_score_ring:
            ring_html = render_fintech_score_ring(score, total, f"{band}")
            st.markdown(f"<div class='fintech-card'>{ring_html}</div>", unsafe_allow_html=True)

        with c_score_desc:
            st.markdown(
                f"""
                <div class="fintech-card">
                    <span class="badge-verified">{band.upper()} PERFORMANCE</span>
                    <h2 style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF; margin: 8px 0 12px 0;">
                        {res.get('recommendation', 'Good Effort')}
                    </h2>
                    <p style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.6;">
                        You answered <b style="color: #FFFFFF;">{score} out of {total}</b> questions correctly ({pct}% Accuracy).
                        Review your mistakes below with verified textbook explanations and core memory hooks.
                    </p>
                    <div style="display: flex; gap: 12px; margin-top: 16px;">
                        <span style="font-size: 0.85rem; color: #94A3B8;">✓ Direct Questions: <b>3</b></span>
                        <span style="font-size: 0.85rem; color: #94A3B8;">✓ Conceptual Questions: <b>4</b></span>
                        <span style="font-size: 0.85rem; color: #94A3B8;">✓ Application Questions: <b>3</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 10 Questions Form
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    with st.form("quiz_form"):
        for q in questions:
            q_id = q.id
            saved_ans = st.session_state.quiz_answers.get(q_id, None)

            # Badge for question type
            type_color = "#3B82F6" if "Direct" in q.type else ("#8B5CF6" if "Conceptual" in q.type else "#10B981")
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.78rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                        QUESTION {q_id} OF 10
                    </span>
                    <span style="background: #1E293B; color: {type_color}; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 6px; border: 1px solid {type_color}66;">
                        {q.type}
                    </span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; line-height: 1.5; margin-bottom: 12px;">
                    {q.question}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Options
            opt_idx = q.options.index(saved_ans) if saved_ans in q.options else None
            user_choice = st.radio(
                f"Select your answer for Q{q_id}",
                options=q.options,
                index=opt_idx,
                key=f"q_radio_{q_id}",
                label_visibility="collapsed",
                disabled=st.session_state.quiz_submitted,
            )
            if user_choice:
                st.session_state.quiz_answers[q_id] = user_choice

            # Post-submission feedback
            if st.session_state.quiz_submitted:
                is_correct = (user_choice == q.answer)
                if is_correct:
                    st.markdown(
                        f"""
                        <div style="background: #064E3B; border: 1.5px solid #059669; border-radius: 12px; padding: 14px 18px; margin: 10px 0 24px 0;">
                            <div style="color: #34D399; font-weight: 800; font-size: 0.95rem;">✓ Correct Answer</div>
                            <div style="color: #D1FAE5; font-size: 0.88rem; margin-top: 4px; line-height: 1.5;">{q.explanation}</div>
                            <div style="color: #A7F3D0; font-size: 0.82rem; font-weight: 700; margin-top: 8px;">🧠 Memory Rule: {q.memory}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background: #450A0A; border: 1.5px solid #DC2626; border-radius: 12px; padding: 14px 18px; margin: 10px 0 24px 0;">
                            <div style="color: #F87171; font-weight: 800; font-size: 0.95rem;">✗ Incorrect (Your choice: {user_choice or 'Unanswered'})</div>
                            <div style="color: #FEE2E2; font-size: 0.9rem; margin-top: 4px; font-weight: 700;">Correct Answer: {q.answer}</div>
                            <div style="color: #FECACA; font-size: 0.88rem; margin-top: 4px; line-height: 1.5;">{q.explanation}</div>
                            <div style="color: #FCA5A5; font-size: 0.82rem; font-weight: 700; margin-top: 8px;">🧠 Memory Rule: {q.memory}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

        col_sub1, col_sub2 = st.columns([2, 1])
        with col_sub1:
            submitted = st.form_submit_button(
                "Submit Answers & View Diagnostic Scoring",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.quiz_submitted,
            )

    if submitted and not st.session_state.quiz_submitted:
        result = grade_quiz_submission(questions, st.session_state.quiz_answers)
        st.session_state.quiz_result = result
        st.session_state.quiz_submitted = True
        st.session_state.student_accuracy = result["percentage"]
        st.rerun()

    # Post-quiz Action Row
    if st.session_state.quiz_submitted:
        c_act1, c_act2, c_act3 = st.columns(3)
        with c_act1:
            if st.button("↺ Retake This Quiz", use_container_width=True):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.session_state.quiz_result = None
                reset_quiz_timer()
                st.rerun()
        with c_act2:
            if st.button("📖 View Book Evidence", use_container_width=True):
                navigate_to("evidence")
        with c_act3:
            if st.button("✦ Analyze Another Topic", type="primary", use_container_width=True):
                navigate_to("analyze")


# ─── PAGE 5: BOOK EVIDENCE ────────────────────────────────────────────────────
elif st.session_state.page == "evidence":
    retrieved_chunks = st.session_state.retrieved_chunks
    if not retrieved_chunks:
        st.info("No book evidence loaded yet. Please run an analysis first.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    st.markdown(
        f"""
        <div style="margin-bottom: 24px;">
            <div class="hero-tag">AUTHORITATIVE SOURCE VERIFICATION</div>
            <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">Authoritative Book Evidence</h1>
            <p style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">
                Direct text segments retrieved from your verified textbooks and past papers for this topic.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"Found <b style='color: #818CF8;'>{len(retrieved_chunks)} verified chunks</b> in curriculum knowledge base:")

    for idx, chunk in enumerate(retrieved_chunks):
        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="badge-exam">{chunk.source_type.upper()}</span>
                        <b style="color: #FFFFFF; font-size: 1.05rem;">{chunk.title}</b>
                    </div>
                    <span style="font-size: 0.8rem; color: #94A3B8;">Page {chunk.page}</span>
                </div>
                <div style="font-size: 0.78rem; font-family: monospace; color: #818CF8; margin-bottom: 12px;">
                    Source ID: {chunk.id}
                </div>
                <div class="source-excerpt">
                    {chunk.text.replace(chr(10), '<br>')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("← Return to Concept Intelligence", type="primary"):
        navigate_to("concept")
