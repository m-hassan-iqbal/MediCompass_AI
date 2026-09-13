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
# End-users cannot see or edit these. Configured via Streamlit Cloud Secrets (or .streamlit/secrets.toml)
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
            <div>• AI Reasoning: {ai_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if chunk_count < 1000:
        with st.expander("ℹ️ Knowledge Base Notice", expanded=False):
            st.markdown(
                """
                <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.4;">
                    Running on verified multi-subject seed chunks. To unlock all <b>7,694 chunks</b> from your Google Drive textbooks on Streamlit Cloud, ensure <code>data/knowledge_base.json</code> is committed and pushed to your GitHub repository.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── PAGE 1: ANALYZE ──────────────────────────────────────────────────────────
if st.session_state.page == "analyze":
    st.markdown(
        """
        <div style="margin-bottom: 28px;">
            <div class="hero-tag">MDCAT • NUMS • Evidence-Based</div>
            <h1 class="hero-title">Study the concept, not the content overload.</h1>
            <p class="hero-sub">
                Enter an exam question or topic. MediCompass retrieves relevant evidence from your configured
                knowledge base and turns it into one clear study decision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_sub, col_exam = st.columns([1, 1])
    with col_sub:
        subject_options = ["Biology", "Chemistry", "Physics"]
        curr_sub_idx = subject_options.index(st.session_state.subject) if st.session_state.subject in subject_options else 0
        selected_subject = st.selectbox("Select Subject", subject_options, index=curr_sub_idx)
        st.session_state.subject = selected_subject

    with col_exam:
        exam_options = ["MDCAT", "NUMS", "MDCAT + NUMS"]
        curr_exam_idx = exam_options.index(st.session_state.exam) if st.session_state.exam in exam_options else 0
        selected_exam = st.selectbox("Select Target Exam", exam_options, index=curr_exam_idx)
        st.session_state.exam = selected_exam

    query_input = st.text_area(
        "Enter Exam Question, Topic, or Concept",
        value=st.session_state.query,
        placeholder="Example:\nWhich type of enzyme inhibition increases Km without changing Vmax?\n\nOr:\nEnzyme inhibition",
        height=130,
        help="You can enter a full multiple-choice question stem, a specific topic, or a short keyword.",
    )

    st.markdown("<div style='font-size: 0.85rem; color: #CBD5E1; margin-bottom: 8px; font-weight: 700;'>Quick test prompts:</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("📌 Newton's Third Law & Motion", use_container_width=True):
        st.session_state.subject = "Physics"
        st.session_state.query = "Why do action and reaction forces according to Newton's third law never cancel each other?"
        st.rerun()
    if c2.button("📌 Enzyme Inhibition & Kinetics", use_container_width=True):
        st.session_state.subject = "Biology"
        st.session_state.query = "Which type of enzyme inhibition increases Km without changing Vmax?"
        st.rerun()
    if c3.button("📌 Periodic Trends & Bonding", use_container_width=True):
        st.session_state.subject = "Chemistry"
        st.session_state.query = "Why does Nitrogen have a higher first ionization energy than Oxygen?"
        st.rerun()

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    if st.button("Analyze with MediCompass →", type="primary", use_container_width=True):
        if not query_input.strip():
            st.error("Please enter a question or topic to analyze.")
        else:
            st.session_state.query = query_input

            progress_placeholder = st.empty()
            status_stages = [
                "Retrieving relevant evidence from Punjab, Federal & Syllabus sources...",
                "Mapping the concept across historical MDCAT/NUMS papers...",
                "Synthesizing source agreements and kinetic definitions...",
                "Building your evidence-based study recommendation...",
            ]

            progress_bar = st.progress(0)
            for step_idx, stage_text in enumerate(status_stages):
                progress_placeholder.markdown(
                    f"""
                    <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 14px 18px; margin: 12px 0;">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #818CF8;">STAGE {step_idx + 1} OF 4</div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-top: 2px;">{stage_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                progress_bar.progress(int((step_idx + 1) * 25))
                time.sleep(0.2)

            if hasattr(vector_store, "retrieve_relevant_chunks"):
                retrieved = vector_store.retrieve_relevant_chunks(
                    query=query_input,
                    subject=st.session_state.subject,
                    exam=st.session_state.exam,
                    top_k=6,
                )
            elif hasattr(vector_store, "search"):
                retrieved = vector_store.search(
                    query=query_input,
                    subject=st.session_state.subject,
                    exam=st.session_state.exam,
                    top_k=6,
                )
            else:
                retrieved = getattr(vector_store, "chunks", [])[:6]
            st.session_state.retrieved_chunks = retrieved

            analysis = groq_client.analyze_concept(
                query=query_input,
                subject=st.session_state.subject,
                exam=st.session_state.exam,
                retrieved_chunks=retrieved,
            )
            st.session_state.analysis = analysis

            quiz_qs = quiz_gen.generate_10_mcq_quiz(
                concept_title=analysis.get("concept_title", f"{st.session_state.subject} Concept"),
                subject=st.session_state.subject,
                exam=st.session_state.exam,
                retrieved_chunks=retrieved,
            )
            st.session_state.quiz_questions = quiz_qs
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_result = None
            reset_quiz_timer()

            progress_placeholder.empty()
            progress_bar.empty()
            navigate_to("concept")


# ─── PAGE 2: CONCEPT INTELLIGENCE ─────────────────────────────────────────────
elif st.session_state.page == "concept":
    analysis = st.session_state.analysis
    if not analysis:
        st.info("No active analysis found. Please start by entering a topic on the Analyze page.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    concept_title = analysis.get("concept_title", "Concept Intelligence")
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
            <div>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span class="badge-exam">{st.session_state.subject}</span>
                    <span class="badge-exam">{st.session_state.exam}</span>
                    <span class="badge-verified">✓ EVIDENCE-GROUNDED</span>
                </div>
                <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">{concept_title}</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analysis.get("api_status") == "error":
        err_msg = analysis.get("api_error", "Connection error")
        st.markdown(
            f"""
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 10px; padding: 10px 14px; margin-bottom: 18px;">
                <div style="font-size: 0.84rem; color: #FDE68A; font-weight: 700;">
                    ⚡ Direct Curriculum Textbook Mode Active (Groq AI Notice: {err_msg})
                </div>
                <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 2px;">
                    Synthesizing evidence-grounded intelligence directly from verified curriculum textbooks and past papers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not groq_ready:
        st.markdown(
            """
            <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 10px 14px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between;">
                <div style="font-size: 0.84rem; color: #E0F2FE;">
                    <b>⚡ Direct Textbook RAG Mode:</b> Grounded directly in your verified curriculum documents.
                </div>
                <div style="font-size: 0.8rem; color: #38BDF8; font-weight: 700;">
                    Add GROQ_API_KEY in Streamlit Secrets for Llama-3.3 dynamic reasoning
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🧠 Quick Memorize Diagram", key="btn_to_diag", use_container_width=True):
            navigate_to("diagram")
    with b2:
        if st.button("🎯 Attempt 10 Focused MCQs", key="btn_to_quiz", type="primary", use_container_width=True):
            navigate_to("quiz")
    with b3:
        if st.button("📖 Visit Exact Book Evidence", key="btn_to_evid", use_container_width=True):
            navigate_to("evidence")

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    col_core, col_priority = st.columns([2.4, 1.2])

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
                <p style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC; line-height: 1.6; margin-bottom: 14px;">
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
            perf_text = "⚠ Personal performance: Not available yet (Take 10-MCQ quiz to calibrate)"
        else:
            perf_text = f"✓ Personal performance: {accuracy}% Accuracy recorded"

        st.markdown(
            f"""
            <div class="fintech-card" style="text-align: center;">
                {ring_svg}
                <div style="font-size: 0.75rem; color: #94A3B8; text-align: left; margin-top: 14px; line-height: 1.5; border-top: 1px solid #1E293B; padding-top: 10px;">
                    <div style="font-weight: 700; margin-bottom: 4px; color: #FFFFFF;">Formula Breakdown:</div>
                    <div>✓ Syllabus weight: 40/40</div>
                    <div>✓ Historical past papers: 35/35</div>
                    <div>{perf_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    deep_pts = analysis.get("deep_explanation", [])
    if deep_pts:
        pts_html = ""
        for pt in deep_pts:
            if "[VERIFIED]" in pt:
                badge = '<span class="badge-verified" style="margin-right: 6px;">VERIFIED</span>'
                clean_pt = pt.replace("[VERIFIED]", "").strip()
            elif "[INFERENCE]" in pt:
                badge = '<span class="badge-inference" style="margin-right: 6px;">INFERENCE</span>'
                clean_pt = pt.replace("[INFERENCE]", "").strip()
            else:
                badge = ""
                clean_pt = pt.strip()
            pts_html += f"<li style='margin-bottom: 10px; line-height: 1.6; font-size: 0.96rem; color: #CBD5E1;'>{badge}{clean_pt}</li>"

        st.markdown(
            f"""
            <div class="fintech-card" style="margin-top: 18px;">
                <div style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 12px;">
                    HIGH-YIELD PRINCIPLES & EXAM TRAPS
                </div>
                <ul style="padding-left: 18px; margin: 0;">
                    {pts_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h3 style='font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 26px 0 14px 0;'>Authoritative Textbook Breakdown</h3>", unsafe_allow_html=True)
    col_ptb, col_fed = st.columns(2)

    with col_ptb:
        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 800; color: #F8FAFC; font-size: 0.95rem;">Punjab Textbook Board (PTB)</span>
                    <span class="badge-verified">OFFICIAL CURRICULUM</span>
                </div>
                <p style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.55; margin-bottom: 8px;">
                    {analysis.get('punjab_synthesis', '')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_fed:
        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 800; color: #F8FAFC; font-size: 0.95rem;">Federal Board (NBF)</span>
                    <span class="badge-verified">OFFICIAL CURRICULUM</span>
                </div>
                <p style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.55; margin-bottom: 8px;">
                    {analysis.get('federal_synthesis', '')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h3 style='font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 26px 0 14px 0;'>Historical MDCAT & NUMS Evidence</h3>", unsafe_allow_html=True)
    past_evid = analysis.get("past_paper_evidence", [])
    if past_evid:
        for ep in past_evid:
            st.markdown(
                f"""
                <div class="past-paper-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 800; color: #FFFFFF; font-size: 0.92rem;">
                            {ep.get('exam', 'MDCAT')} {ep.get('year', 'Recent')}
                        </span>
                        <span class="badge-verified">VERIFIED EXAM OUTCOME</span>
                    </div>
                    <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.5;">
                        {ep.get('summary', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("← Analyze Another Topic", use_container_width=True):
            navigate_to("analyze")
    with c_btn2:
        if st.button("Test This Concept (10 MCQs) →", type="primary", use_container_width=True):
            navigate_to("quiz")


# ─── PAGE 3: QUICK DIAGRAM ───────────────────────────────────────────────────
elif st.session_state.page == "diagram":
    analysis = st.session_state.analysis
    if not analysis:
        st.info("No concept diagram available. Please analyze a question first.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    concept_title = analysis.get("concept_title", "Concept")
    diag = analysis.get("diagram", {})
    quick_recall = analysis.get("quick_recall", [])

    st.markdown(
        f"""
        <div style="margin-bottom: 22px;">
            <div class="hero-tag">Active Recall • 60-Second Mastery</div>
            <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">Quick Memorize Diagram</h1>
            <p style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">
                Visual relationship model for <b>{concept_title}</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_graph, col_recall = st.columns([1.8, 1.2])

    with col_graph:
        st.markdown(
            """
            <div class="fintech-card" style="margin-bottom: 16px;">
                <div style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;">
                    CONCEPT INTERACTION MAP
                </div>
            """,
            unsafe_allow_html=True,
        )

        svg_diagram = render_concept_diagram_html(
            title=diag.get("title", f"{concept_title} Cascade"),
            nodes=diag.get("nodes", []),
            connections=diag.get("connections", []),
        )
        st.markdown(svg_diagram, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_recall:
        recall_cards = ""
        for idx, item in enumerate(quick_recall):
            recall_cards += f"""
            <div style="background: #080B11; border: 1px solid #1E293B; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;">
                <div style="font-size: 0.72rem; font-weight: 800; color: #818CF8; letter-spacing: 0.06em;">POCKET RULE {idx+1}</div>
                <div style="font-size: 0.92rem; font-weight: 600; color: #F8FAFC; margin-top: 4px; line-height: 1.4;">{item}</div>
            </div>
            """

        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 12px;">
                    60-SECOND QUICK RECALL
                </div>
                <h3 style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 14px;">
                    {diag.get('memory_hook', 'Core Concept Rule')}
                </h3>
                {recall_cards}
            </div>
            """,
            unsafe_allow_html=True,
        )

    c_diag1, c_diag2 = st.columns(2)
    with c_diag1:
        if st.button("← Return to Concept Intelligence", use_container_width=True):
            navigate_to("concept")
    with c_diag2:
        if st.button("🎯 Test Concept with 10 MCQs →", type="primary", use_container_width=True):
            navigate_to("quiz")


# ─── PAGE 4: 10-MCQ QUIZ ─────────────────────────────────────────────────────
elif st.session_state.page == "quiz":
    questions = st.session_state.quiz_questions
    if not questions:
        st.info("No quiz generated yet. Run an analysis to generate a personalized 10-MCQ Concept Check.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    if st.session_state.quiz_submitted and st.session_state.quiz_result:
        result = st.session_state.quiz_result

        st.markdown(
            """
            <div style="margin-bottom: 20px;">
                <div class="hero-tag">CONCEPT MASTERY SCORECARD</div>
                <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">Evaluation Results</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_score, col_feedback = st.columns([1, 2])

        with col_score:
            score_ring = render_fintech_score_ring(result["score"], result["total"])
            st.markdown(
                f"""
<div class="fintech-card" style="text-align: center;">
    {score_ring}
    <div style="margin-top: 10px;">
        <span style="background: {result['color_theme']}1A; color: {result['color_theme']}; font-weight: 800; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; border: 1px solid {result['color_theme']}33;">
            {result['mastery_label']}
        </span>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with col_feedback:
            st.markdown(
                f"""
<div class="fintech-card">
    <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
        RECOMMENDED STUDY ACTION
    </span>
    <h3 style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 8px 0;">
        {result['mastery_label']}
    </h3>
    <p style="font-size: 1.02rem; color: #CBD5E1; line-height: 1.6;">
        {result['action_message']}
    </p>
    <div style="margin-top: 16px; padding: 12px 16px; background: #080B11; border: 1px solid #1E293B; border-radius: 12px; font-size: 0.88rem; color: #94A3B8;">
        Your performance score has been calibrated into your <b>Evidence Study Priority</b>.
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("<h3 style='font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 24px 0 16px 0;'>Detailed Question Breakdown</h3>", unsafe_allow_html=True)

        for item in result["detailed_results"]:
            q_id = item["id"]
            is_correct = item["is_correct"]
            status_symbol = "✓ Correct" if is_correct else "✕ Incorrect"
            status_color = "#4ADE80" if is_correct else "#F87171"
            border_color = "rgba(74, 222, 128, 0.4)" if is_correct else "rgba(248, 113, 113, 0.4)"
            bg_color = "rgba(74, 222, 128, 0.12)" if is_correct else "rgba(248, 113, 113, 0.12)"

            st.markdown(
                f"""
                <div class="fintech-card" style="border-left: 5px solid {status_color}; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 800; font-size: 0.85rem; color: #94A3B8;">QUESTION {q_id} • {item['type']}</span>
                        <span style="background: {bg_color}; color: {status_color}; font-weight: 800; font-size: 0.78rem; padding: 3px 10px; border-radius: 9999px; border: 1px solid {border_color};">
                            {status_symbol}
                        </span>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">
                        {item['question']}
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
                        <div style="padding: 8px 12px; background: #080B11; border: 1px solid #1E293B; border-radius: 8px; font-size: 0.88rem;">
                            <span style="color: #94A3B8; font-weight: 600;">Your choice:</span>
                            <b style="color: {'#4ADE80' if is_correct else '#F87171'}; margin-left: 6px;">{item['user_answer']}</b>
                        </div>
                        <div style="padding: 8px 12px; background: rgba(74, 222, 128, 0.08); border: 1px solid rgba(74, 222, 128, 0.25); border-radius: 8px; font-size: 0.88rem;">
                            <span style="color: #4ADE80; font-weight: 600;">Correct answer:</span>
                            <b style="color: #4ADE80; margin-left: 6px;">{item['correct_answer']}</b>
                        </div>
                    </div>
                    <div style="background: #080B11; border: 1px solid #1E293B; border-radius: 10px; padding: 12px 14px; font-size: 0.9rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 8px;">
                        <b style="color: #818CF8;">Deep Concept:</b> {item['explanation']}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: #94A3B8; padding-top: 6px;">
                        <span>🧠 <b style="color: #CBD5E1;">Remember:</b> {item['memory']}</span>
                        <span>📌 {item['past_paper']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_retake, col_back = st.columns(2)
        with col_retake:
            if st.button("↺ Retake 10-MCQ Concept Check", use_container_width=True):
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_result = None
                reset_quiz_timer()
                st.rerun()
        with col_back:
            if st.button("← Return to Concept Intelligence", type="primary", use_container_width=True):
                navigate_to("concept")

    else:
        remaining_sec, timer_str, is_expired = get_quiz_timer_state(total_seconds=600)

        col_hdr, col_timer = st.columns([3, 1])
        with col_hdr:
            st.markdown(
                """
                <div>
                    <div class="hero-tag">MDCAT & NUMS • TIMED CONCEPT CHECK</div>
                    <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">10-MCQ Concept Check</h1>
                    <p style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">
                        Test whether you understand the concept, not whether you memorized an old question.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_timer:
            timer_color = "#F87171" if remaining_sec < 120 else "#818CF8"
            st.markdown(
                f"""
                <div style="background: #0F172A; border: 2px solid {timer_color}; border-radius: 16px; padding: 12px; text-align: center; box-shadow: 0 4px 14px rgba(0,0,0,0.5);">
                    <div style="font-size: 0.72rem; font-weight: 800; color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase;">TIME REMAINING</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {timer_color}; font-variant-numeric: tabular-nums;">
                        {timer_str}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if is_expired:
            st.warning("⏰ Time has expired for this 10-minute session! Please submit your responses.")

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        answered_count = sum(1 for q in questions if q.id in st.session_state.quiz_answers)
        st.progress(answered_count / len(questions))
        st.caption(f"Progress: {answered_count} of {len(questions)} questions answered")

        for q in questions:
            selected_val = st.session_state.quiz_answers.get(q.id, None)
            idx_in_options = None
            if selected_val in q.options:
                idx_in_options = q.options.index(selected_val)

            st.markdown(
                f"""
                <div class="fintech-card" style="margin-bottom: 14px; padding-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 0.78rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                            QUESTION {q.id} OF 10 • {q.type}
                        </span>
                        <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">MDCAT / NUMS Format</span>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; line-height: 1.5; margin-bottom: 12px;">
                        {q.question}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            choice = st.radio(
                f"Question {q.id} Choices",
                options=q.options,
                index=idx_in_options,
                key=f"q_radio_{q.id}",
                label_visibility="collapsed",
            )

            if choice:
                st.session_state.quiz_answers[q.id] = choice

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        if st.button("Submit Concept Check for Grading →", type="primary", use_container_width=True):
            result = grade_quiz_submission(questions, st.session_state.quiz_answers)
            st.session_state.quiz_result = result
            st.session_state.quiz_submitted = True
            st.session_state.student_accuracy = result["percentage"]

            if st.session_state.analysis:
                old_score = st.session_state.analysis.get("priority_score", 80)
                calibrated = max(40, min(95, int(old_score * 0.7 + (100 - result["percentage"]) * 0.3)))
                st.session_state.analysis["priority_score"] = calibrated
                if calibrated >= 75:
                    st.session_state.analysis["priority_label"] = "STUDY NOW"
                elif calibrated >= 50:
                    st.session_state.analysis["priority_label"] = "REVIEW SOON"
                else:
                    st.session_state.analysis["priority_label"] = "LOWER PRIORITY"

            st.rerun()


# ─── PAGE 5: BOOK EVIDENCE ───────────────────────────────────────────────────
elif st.session_state.page == "evidence":
    retrieved = st.session_state.retrieved_chunks
    analysis = st.session_state.analysis

    st.markdown(
        """
        <div style="margin-bottom: 22px;">
            <div class="hero-tag">Verified Curriculum Traceability</div>
            <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">Authoritative Book Evidence</h1>
            <p style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">
                Direct text segments retrieved from your verified textbooks and past papers for this topic.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not retrieved:
        st.info("No retrieved document evidence in active session. Please start on the Analyze page.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    st.markdown(f"<div style='font-size: 0.9rem; color: #CBD5E1; margin-bottom: 16px;'>Found <b>{len(retrieved)} verified chunks</b> in curriculum knowledge base:</div>", unsafe_allow_html=True)

    for i, chunk in enumerate(retrieved):
        badge_style = "background: rgba(129, 140, 248, 0.15); color: #818CF8; border: 1px solid rgba(129, 140, 248, 0.3);"
        if chunk.source_type == "Punjab Book":
            badge_style = "background: rgba(34, 197, 94, 0.15); color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.3);"
        elif chunk.source_type == "Federal Book":
            badge_style = "background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3);"
        elif chunk.source_type == "Past Paper":
            badge_style = "background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3);"

        page_info = f"Page {chunk.page}" if chunk.page and chunk.page != "unknown" else "Page N/A"

        st.markdown(
            f"""
            <div class="fintech-card" style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="{badge_style} font-weight: 800; font-size: 0.75rem; padding: 2px 10px; border-radius: 9999px;">
                            {chunk.source_type.upper()}
                        </span>
                        <span style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem;">{chunk.title}</span>
                    </div>
                    <span style="font-size: 0.8rem; color: #94A3B8; font-weight: 600;">{page_info}</span>
                </div>
                <div style="font-size: 0.8rem; color: #818CF8; font-family: monospace; margin-bottom: 10px;">
                    Source ID: {chunk.id}
                </div>
                <div style="background: #080B11; border: 1px solid #1E293B; border-radius: 10px; padding: 14px; font-size: 0.92rem; color: #CBD5E1; line-height: 1.6; white-space: pre-wrap;">
{chunk.text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("← Return to Concept Intelligence", use_container_width=True):
        navigate_to("concept")
