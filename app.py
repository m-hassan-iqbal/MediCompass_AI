"""
MediCompass AI - Master Streamlit Application
Turning exam information into study intelligence for MDCAT and NUMS aspirants.
"""

import os
import time
import logging
import streamlit as st
import streamlit.components.v1 as components

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
    clean_html,
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

# Initialize Session State & Inject Fintech CSS Styles
init_session_state()
apply_custom_styles()

# ─── APPLICATION SECRETS CONFIGURATION (Streamlit Cloud & Local Safe) ───────
GROQ_API_KEY = ""
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
elif os.environ.get("GROQ_API_KEY"):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

raw_model = ""
if "GROQ_MODEL" in st.secrets:
    raw_model = st.secrets["GROQ_MODEL"]
elif os.environ.get("GROQ_MODEL"):
    raw_model = os.environ.get("GROQ_MODEL")

if not raw_model or raw_model in ["llama-3.1-8b-instant", "llama3.1-8b"]:
    GROQ_MODEL = "llama-3.3-70b-versatile"
else:
    GROQ_MODEL = raw_model

DEFAULT_GDRIVE_URL = "https://drive.google.com/drive/folders/12JMNPPtw9ranu47PpEOFTqupACOHO66q?usp=sharing"
if "GDRIVE_URL" in st.secrets:
    GDRIVE_URL = st.secrets["GDRIVE_URL"]
elif "GOOGLE_DRIVE_FOLDER_URL" in st.secrets:
    GDRIVE_URL = st.secrets["GOOGLE_DRIVE_FOLDER_URL"]
elif os.environ.get("GDRIVE_URL"):
    GDRIVE_URL = os.environ.get("GDRIVE_URL")
else:
    GDRIVE_URL = DEFAULT_GDRIVE_URL

# ─── CACHED VECTOR STORE INITIALIZATION ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_initialized_vector_store():
    """
    Initializes knowledge base by syncing Google Drive and indexing local/downloaded PDFs.
    Cached across sessions to provide lightning-fast, zero-rebuild startup.
    """
    sources_dir = "data/sources"
    manifest_path = "data/source_manifest.json"
    past_papers_path = "data/past_papers.json"

    # Sync Google Drive folder if configured
    if GDRIVE_URL and not GDRIVE_URL.startswith("#"):
        try:
            sync_google_drive_public_folders([GDRIVE_URL], sources_dir)
        except Exception as e:
            logger.warning(f"Google Drive sync skipped or encountered error: {e}")

    # Compute fingerprint to determine if cache can be reused
    fingerprint = compute_kb_fingerprint(sources_dir, manifest_path)
    vector_store = LocalVectorStore(cache_dir="cache")

    if not vector_store.load_cached_index(fingerprint):
        logger.info(f"Rebuilding index for fingerprint: {fingerprint}")
        chunks = build_knowledge_base_chunks(
            sources_dir=sources_dir,
            manifest_path=manifest_path,
            past_papers_path=past_papers_path,
        )
        vector_store.build_index(chunks, fingerprint)

    return vector_store, fingerprint


# Load Vector Store
vector_store, current_fingerprint = get_initialized_vector_store()

# Initialize API Clients
groq_client = GroqClient(api_key=GROQ_API_KEY, model=GROQ_MODEL)
quiz_generator = QuizGenerator(api_key=GROQ_API_KEY, model=GROQ_MODEL)


# ─── SIDEBAR NAVIGATION & FINTECH METRICS ───────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 20px 0;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.5rem; color: #818CF8;">✦</span>
                <span style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; color: #FFFFFF;">MediCompass</span>
            </div>
            <div style="font-size: 0.72rem; font-weight: 800; color: #818CF8; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px;">
                AI • EXAM INTELLIGENCE
            </div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px; line-height: 1.4;">
                Turning exam information into study intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-section-title'>NAVIGATION</div>", unsafe_allow_html=True)

    nav_pages = [
        ("analyze", "⌂ Analyze", "Target new high-yield concept"),
        ("concept", "◈ Concept Intelligence", "Read verified synthesis"),
        ("diagram", "🧠 Quick Diagram", "Visual concept flowchart"),
        ("quiz", "✓ 10-MCQ Quiz", "Calibrated concept mastery check"),
        ("evidence", "📖 Book Evidence", "Authoritative source excerpts"),
    ]

    for page_id, label, desc in nav_pages:
        is_active = st.session_state.page == page_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_{page_id}", type=btn_type, use_container_width=True):
            navigate_to(page_id)

    st.markdown("---")

    # Current Target Status Card
    st.markdown("<div class='sidebar-section-title'>CURRENT TARGET</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background: #0F172A; border: 1.5px solid #1E293B; border-radius: 14px; padding: 14px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">Subject</span>
                <span style="font-size: 0.75rem; color: #818CF8; font-weight: 800;">{st.session_state.subject}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">Target Exam</span>
                <span style="font-size: 0.75rem; color: #60A5FA; font-weight: 800;">{st.session_state.exam}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Knowledge Base Engine Status
    st.markdown("<div class='sidebar-section-title'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    num_chunks = len(vector_store.chunks)
    has_groq = groq_client.is_configured()

    st.markdown(
        f"""
        <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.8;">
            <div>• Knowledge Base: <b style="color: #FFFFFF;">{num_chunks} Multi-Subject chunks</b></div>
            <div>• Official books: <b style="color: #FFFFFF;">Punjab & Federal (PTB, NBF)</b></div>
            <div>• Past Papers: <b style="color: #FFFFFF;">MDCAT & NUMS 2020-24</b></div>
            <div>• RAG Engine: <b style="color: {'#4ADE80' if has_groq else '#F59E0B'};">{'Active Groq (' + GROQ_MODEL + ')' if has_groq else 'Grounded Textbook Mode'}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_groq:
        st.markdown(
            """
            <div style="margin-top: 10px; padding: 10px 12px; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; font-size: 0.76rem; color: #F59E0B; line-height: 1.4;">
                ✦ Running in <b>Authoritative Multi-Subject Mode</b>. Add your <code>GROQ_API_KEY</code> in Streamlit Secrets to enable open-ended generation.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── PAGE 1: ANALYZE / SEARCH ────────────────────────────────────────────────
if st.session_state.page == "analyze":
    st.markdown(
        """
        <div class="hero-tag">EXAM INTELLIGENCE ENGINE</div>
        <h1 class="hero-title">Master Any High-Yield Concept for MDCAT & NUMS</h1>
        <p class="hero-sub">
            Query across authoritative Punjab textbooks, Federal curriculum, PMDC syllabus,
            and verified past paper appearances with zero hallucinations.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.form("search_form"):
        col_q, col_sub, col_ex = st.columns([3, 1, 1])

        with col_q:
            query_input = st.text_input(
                "Enter concept, topic, or question",
                value=st.session_state.query,
                placeholder="e.g. Competitive vs Non-Competitive Enzyme Inhibition, or Newton's Third Law",
            )

        with col_sub:
            subject_choice = st.selectbox(
                "Subject",
                ["Biology", "Chemistry", "Physics"],
                index=["Biology", "Chemistry", "Physics"].index(st.session_state.subject),
            )

        with col_ex:
            exam_choice = st.selectbox(
                "Target Exam",
                ["MDCAT", "NUMS", "MDCAT + NUMS"],
                index=["MDCAT", "NUMS", "MDCAT + NUMS"].index(st.session_state.exam),
            )

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        btn_submit = st.form_submit_button("Generate Concept Intelligence →", type="primary", use_container_width=True)

    if btn_submit:
        st.session_state.subject = subject_choice
        st.session_state.exam = exam_choice
        st.session_state.query = query_input

        with st.spinner("Retrieving authoritative textbooks, syllabus outcomes, and past papers..."):
            retrieved = vector_store.retrieve_relevant_chunks(
                query=query_input,
                subject=subject_choice,
                exam=exam_choice,
                top_k=6,
            )
            st.session_state.retrieved_chunks = retrieved

            # Generate Analysis via Groq (with grounded fallback)
            analysis_data = groq_client.analyze_concept(
                query=query_input,
                subject=subject_choice,
                exam=exam_choice,
                retrieved_chunks=retrieved,
            )
            st.session_state.analysis = analysis_data

            # Generate or load aligned 10-MCQ quiz
            questions = quiz_generator.generate_10_mcq_quiz(
                concept_title=analysis_data.get("concept_title", query_input),
                subject=subject_choice,
                exam=exam_choice,
                retrieved_chunks=retrieved,
            )
            st.session_state.quiz_questions = questions
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_result = None
            reset_quiz_timer()

            navigate_to("concept")

    # High-Yield Exam Concept Cards for Fast Prototyping
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-section-title" style="margin-bottom: 14px;">
            HIGH-FREQUENCY MDCAT & NUMS CONCEPTS
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="fintech-card">
                <span class="badge-exam">BIOLOGY • MDCAT</span>
                <h3 style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin: 10px 0 6px 0;">
                    Competitive vs Non-Competitive Enzyme Inhibition
                </h3>
                <p style="font-size: 0.86rem; color: #94A3B8; line-height: 1.5;">
                    Vmax and Km kinetic shifts, active site competition, and sulfa drug pharmacology.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Enzyme Inhibition →", key="btn_q1", use_container_width=True):
            st.session_state.subject = "Biology"
            st.session_state.exam = "MDCAT"
            st.session_state.query = "Competitive vs Non-Competitive Enzyme Inhibition"
            retrieved = vector_store.retrieve_relevant_chunks(
                query=st.session_state.query, subject="Biology", exam="MDCAT", top_k=6
            )
            st.session_state.retrieved_chunks = retrieved
            st.session_state.analysis = groq_client.analyze_concept(
                st.session_state.query, "Biology", "MDCAT", retrieved
            )
            st.session_state.quiz_questions = quiz_generator.generate_10_mcq_quiz(
                "Competitive Enzyme Inhibition", "Biology", "MDCAT", retrieved
            )
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_result = None
            reset_quiz_timer()
            navigate_to("concept")

    with c2:
        st.markdown(
            """
            <div class="fintech-card">
                <span class="badge-exam">PHYSICS • MDCAT</span>
                <h3 style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin: 10px 0 6px 0;">
                    Newton's 3rd Law & Momentum Conservation
                </h3>
                <p style="font-size: 0.86rem; color: #94A3B8; line-height: 1.5;">
                    Why action and reaction pairs never cancel, rocket recoil dynamics, and collisions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Newton's Laws →", key="btn_q2", use_container_width=True):
            st.session_state.subject = "Physics"
            st.session_state.exam = "MDCAT"
            st.session_state.query = "Newton's Third Law and Action-Reaction Pairs"
            retrieved = vector_store.retrieve_relevant_chunks(
                query=st.session_state.query, subject="Physics", exam="MDCAT", top_k=6
            )
            st.session_state.retrieved_chunks = retrieved
            st.session_state.analysis = groq_client.analyze_concept(
                st.session_state.query, "Physics", "MDCAT", retrieved
            )
            st.session_state.quiz_questions = quiz_generator.generate_10_mcq_quiz(
                "Newton's Third Law of Motion", "Physics", "MDCAT", retrieved
            )
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_result = None
            reset_quiz_timer()
            navigate_to("concept")

    with c3:
        st.markdown(
            """
            <div class="fintech-card">
                <span class="badge-exam">CHEMISTRY • NUMS</span>
                <h3 style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin: 10px 0 6px 0;">
                    Periodic Trends & Ionization Energy Anomalies
                </h3>
                <p style="font-size: 0.86rem; color: #94A3B8; line-height: 1.5;">
                    Nuclear charge vs shielding, nitrogen half-filled 2p3 subshell, and coordinate bonds.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Periodic Trends →", key="btn_q3", use_container_width=True):
            st.session_state.subject = "Chemistry"
            st.session_state.exam = "NUMS"
            st.session_state.query = "Periodic Trends and Ionization Energy Anomalies"
            retrieved = vector_store.retrieve_relevant_chunks(
                query=st.session_state.query, subject="Chemistry", exam="NUMS", top_k=6
            )
            st.session_state.retrieved_chunks = retrieved
            st.session_state.analysis = groq_client.analyze_concept(
                st.session_state.query, "Chemistry", "NUMS", retrieved
            )
            st.session_state.quiz_questions = quiz_generator.generate_10_mcq_quiz(
                "Periodic Trends and Chemical Bonding", "Chemistry", "NUMS", retrieved
            )
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_result = None
            reset_quiz_timer()
            navigate_to("concept")


# ─── PAGE 2: CONCEPT INTELLIGENCE ────────────────────────────────────────────
elif st.session_state.page == "concept":
    analysis = st.session_state.analysis
    if not analysis:
        st.info("No active concept analysis found. Please run an analysis first.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    # Header section
    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            <div class="hero-tag">{st.session_state.subject.upper()} • {st.session_state.exam} CURRICULUM</div>
            <h1 style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin: 0;">
                {analysis.get('concept_title', 'Concept Intelligence')}
            </h1>
            <p style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">
                {analysis.get('question_summary', '')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Secondary Quick Action Buttons
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🧠 View Quick Memorize Diagram", key="btn_to_diag", use_container_width=True):
            navigate_to("diagram")
    with b2:
        if st.button("🎯 Attempt 10 Focused MCQs", key="btn_to_quiz", type="primary", use_container_width=True):
            navigate_to("quiz")
    with b3:
        if st.button("📖 Visit Exact Book Evidence", key="btn_to_evid", use_container_width=True):
            navigate_to("evidence")

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    # Top Grid: Core Concept Card + Study Priority Ring
    col_core, col_priority = st.columns([2.4, 1.2])

    with col_core:
        core_card = (
            f'<div class="fintech-card">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">'
            f'<span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">CORE CONCEPT</span>'
            f'<span class="badge-verified">VERIFIED</span>'
            f'</div>'
            f'<p style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC; line-height: 1.6; margin-bottom: 14px;">{analysis.get("core_explanation", "")}</p>'
            f'<div class="memory-box">'
            f'<div style="font-size: 0.75rem; font-weight: 800; color: #818CF8; text-transform: uppercase; letter-spacing: 0.05em;">REMEMBER IT FAST</div>'
            f'<div style="font-size: 1.02rem; font-weight: 800; color: #FFFFFF; margin-top: 4px;">{analysis.get("memory_hook", "")}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(clean_html(core_card), unsafe_allow_html=True)

    with col_priority:
        p_score = analysis.get("priority_score", 85)
        p_label = analysis.get("priority_label", "STUDY NOW")
        ring_svg = render_fintech_priority_ring(p_score, p_label)

        # Dynamic formula explanation based on whether student has attempted the quiz
        accuracy = st.session_state.student_accuracy
        if accuracy is None:
            perf_text = "⚠ Personal performance: Not available yet (Take 10-MCQ quiz to calibrate)"
        else:
            perf_text = f"✓ Personal performance: {accuracy}% Accuracy recorded"

        priority_card = (
            f'<div class="fintech-card" style="text-align: center;">'
            f'{ring_svg}'
            f'<div style="font-size: 0.75rem; color: #94A3B8; text-align: left; margin-top: 14px; line-height: 1.5; border-top: 1px solid #1E293B; padding-top: 10px;">'
            f'<div style="font-weight: 700; margin-bottom: 4px; color: #FFFFFF;">Formula Breakdown:</div>'
            f'<div>✓ Syllabus weight: 40/40</div>'
            f'<div>✓ Historical past papers: 35/35</div>'
            f'<div>{perf_text}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(clean_html(priority_card), unsafe_allow_html=True)

    # Deep Concept Explanation
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
            <div class="fintech-card">
                <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                    DEEP CONCEPT EXPLANATION & EXAM TRAPS
                </span>
                <ul style="margin: 14px 0 0 0; padding-left: 20px;">
                    {pts_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Inter-Board Comparison Section (Punjab vs Federal vs Syllabus)
    st.markdown(
        """
        <div style="margin: 28px 0 14px 0;">
            <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                MULTI-BOARD SYNTHESIS
            </span>
            <h3 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 0 0;">
                Punjab vs Federal Board Comparison
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_pb, c_fb = st.columns(2)
    with c_pb:
        st.markdown(
            f"""
            <div class="fintech-card" style="height: 100%;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                    <span style="font-size: 0.82rem; font-weight: 800; color: #60A5FA;">📘 PUNJAB TEXTBOOK (PTB)</span>
                    <span class="badge-verified">AUTHORITATIVE</span>
                </div>
                <p style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.6;">
                    {analysis.get('punjab_synthesis', '').replace('[VERIFIED]', '').strip()}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_fb:
        st.markdown(
            f"""
            <div class="fintech-card" style="height: 100%;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                    <span style="font-size: 0.82rem; font-weight: 800; color: #34D399;">📗 FEDERAL TEXTBOOK (NBF)</span>
                    <span class="badge-verified">AUTHORITATIVE</span>
                </div>
                <p style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.6;">
                    {analysis.get('federal_synthesis', '').replace('[VERIFIED]', '').strip()}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Synthesis Takeaway
    st.markdown(
        f"""
        <div style="margin-top: 14px; background: #080B11; border: 1px solid #1E293B; border-radius: 14px; padding: 16px 20px;">
            <b style="color: #818CF8; font-size: 0.9rem;">Synthesis Takeaway:</b>
            <span style="color: #CBD5E1; font-size: 0.92rem; margin-left: 8px;">
                {analysis.get('synthesis_takeaway', '')}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Historical Past Paper Concept Signal
    st.markdown(
        """
        <div style="margin: 28px 0 14px 0;">
            <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                HISTORICAL EVIDENCE TRACKER
            </span>
            <h3 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 0 0;">
                Verified Past Paper Appearances
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    past_papers = analysis.get("past_paper_evidence", [])
    if past_papers:
        for p in past_papers:
            st.markdown(
                f"""
                <div class="fintech-card" style="padding: 16px 20px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span class="badge-exam">{p.get('exam', 'MDCAT')} {p.get('year', '')}</span>
                        <span style="font-size: 0.95rem; color: #FFFFFF; font-weight: 600; margin-left: 8px;">
                            {p.get('summary', '').replace('[VERIFIED]', '').strip()}
                        </span>
                    </div>
                    <span class="badge-verified">VERIFIED PAPER LOG</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No direct past paper appearances logged for this concept yet.")


# ─── PAGE 3: QUICK MEMORIZE DIAGRAM ──────────────────────────────────────────
elif st.session_state.page == "diagram":
    analysis = st.session_state.analysis
    if not analysis or "diagram" not in analysis:
        st.info("No diagram generated for this concept yet. Please run an analysis first.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="hero-tag">COGNITIVE COMPRESSION • RAPID RECALL</div>
            <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">Quick Memorize Diagram</h1>
            <p style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">
                Visualizing cause-and-effect relationships allows instant mental recall during high-pressure medical entrance exams.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render interactive concept flowchart
    diagram_html = render_concept_diagram_html(analysis.get("diagram", {}))
    st.markdown(clean_html(diagram_html), unsafe_allow_html=True)

    # Quick Recall Points
    quick_recall = analysis.get("quick_recall", [])
    if quick_recall:
        recall_cards = ""
        for pt in quick_recall:
            recall_cards += (
                f'<div style="background: #080B11; border: 1px solid #1E293B; border-radius: 14px; padding: 14px 18px; margin-bottom: 10px; display: flex; align-items: center; gap: 12px;">'
                f'<span style="color: #818CF8; font-size: 1.2rem; font-weight: 800;">✓</span>'
                f'<span style="font-size: 0.95rem; color: #FFFFFF; font-weight: 600;">{pt}</span>'
                f'</div>'
            )

        st.markdown(
            clean_html(
                f'<div class="fintech-card">'
                f'<span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">RAPID EXAM RETRIEVAL CUES</span>'
                f'<h3 style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 16px 0;">Key Recall Anchors</h3>'
                f'{recall_cards}'
                f'</div>'
            ),
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


# ─── PAGE 4: 10-MCQ QUIZ ─────────────────────────────────────────────────────
elif st.session_state.page == "quiz":
    questions = st.session_state.quiz_questions
    if not questions:
        st.info("No quiz generated yet. Run an analysis to generate a personalized 10-MCQ Concept Check.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    # If already submitted, show results view
    if st.session_state.quiz_submitted and st.session_state.quiz_result:
        result = st.session_state.quiz_result
        score_val = result.get("score", 0)
        total_val = result.get("total", 10)
        mastery_lbl = result.get("mastery_band") or result.get("mastery_label") or "Good foundation"
        color_theme = result.get("color_theme", "#818CF8")
        action_msg = result.get("action_message", "Review the missed questions and keep the core rule active.")
        detailed_results = result.get("detailed_results", [])

        header_html = (
            f'<div style="margin-bottom: 20px;">'
            f'<div class="hero-tag">EVALUATION • DIAGNOSTIC SUMMARY</div>'
            f'<h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">10-MCQ Concept Check Results</h1>'
            f'</div>'
        )
        st.markdown(clean_html(header_html), unsafe_allow_html=True)

        col_score, col_feedback = st.columns([1.2, 2.4])
        with col_score:
            score_ring = render_fintech_score_ring(score_val, total_val)
            score_card = (
                f'<div class="fintech-card" style="text-align: center;">'
                f'{score_ring}'
                f'<div style="margin-top: 10px;">'
                f'<span style="background: {color_theme}1A; color: {color_theme}; font-weight: 800; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; border: 1px solid {color_theme}33;">'
                f'{mastery_lbl}'
                f'</span>'
                f'</div>'
                f'</div>'
            )
            st.markdown(clean_html(score_card), unsafe_allow_html=True)

        with col_feedback:
            feedback_card = (
                f'<div class="fintech-card">'
                f'<span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">RECOMMENDED STUDY ACTION</span>'
                f'<h3 style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 8px 0;">{mastery_lbl}</h3>'
                f'<p style="font-size: 1.02rem; color: #CBD5E1; line-height: 1.6;">{action_msg}</p>'
                f'<div style="margin-top: 16px; padding: 12px 16px; background: #080B11; border: 1px solid #1E293B; border-radius: 12px; font-size: 0.88rem; color: #94A3B8;">'
                f'Your performance score has been calibrated into your <b>Evidence Study Priority</b>.'
                f'</div>'
                f'</div>'
            )
            st.markdown(clean_html(feedback_card), unsafe_allow_html=True)

        # Question Review Section
        st.markdown(clean_html("<h3 style='font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 24px 0 16px 0;'>Detailed Question Breakdown</h3>"), unsafe_allow_html=True)

        for item in detailed_results:
            q_id = item.get("id", 1)
            is_correct = item.get("is_correct", False)
            status_symbol = "✓ Correct" if is_correct else "✕ Incorrect"
            status_color = "#4ADE80" if is_correct else "#F87171"
            border_color = "rgba(74, 222, 128, 0.4)" if is_correct else "rgba(248, 113, 113, 0.4)"
            bg_color = "rgba(74, 222, 128, 0.12)" if is_correct else "rgba(248, 113, 113, 0.12)"

            q_card = (
                f'<div class="fintech-card" style="border-left: 5px solid {status_color}; margin-bottom: 16px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
                f'<span style="font-weight: 800; font-size: 0.85rem; color: #94A3B8;">QUESTION {q_id} • {item.get("type", "Concept")}</span>'
                f'<span style="background: {bg_color}; color: {status_color}; font-weight: 800; font-size: 0.78rem; padding: 3px 10px; border-radius: 9999px; border: 1px solid {border_color};">{status_symbol}</span>'
                f'</div>'
                f'<div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">{item.get("question", "")}</div>'
                f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">'
                f'<div style="padding: 8px 12px; background: #080B11; border: 1px solid #1E293B; border-radius: 8px; font-size: 0.88rem;">'
                f'<span style="color: #94A3B8; font-weight: 600;">Your choice:</span>'
                f'<b style="color: {"#4ADE80" if is_correct else "#F87171"}; margin-left: 6px;">{item.get("user_answer", "Unanswered")}</b>'
                f'</div>'
                f'<div style="padding: 8px 12px; background: rgba(74, 222, 128, 0.08); border: 1px solid rgba(74, 222, 128, 0.25); border-radius: 8px; font-size: 0.88rem;">'
                f'<span style="color: #4ADE80; font-weight: 600;">Correct answer:</span>'
                f'<b style="color: #4ADE80; margin-left: 6px;">{item.get("correct_answer", "")}</b>'
                f'</div>'
                f'</div>'
                f'<div style="background: #080B11; border: 1px solid #1E293B; border-radius: 10px; padding: 12px 14px; font-size: 0.9rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 8px;">'
                f'<b style="color: #818CF8;">Deep Concept:</b> {item.get("explanation", "")}'
                f'</div>'
                f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: #94A3B8; padding-top: 6px;">'
                f'<span>🧠 <b style="color: #CBD5E1;">Remember:</b> {item.get("memory", "")}</span>'
                f'<span>📌 {item.get("past_paper", "")}</span>'
                f'</div>'
                f'</div>'
            )
            st.markdown(clean_html(q_card), unsafe_allow_html=True)

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
        # Live Quiz Taking Experience
        remaining_sec, timer_str, is_expired = get_quiz_timer_state(total_seconds=600)

        # Header with live countdown timer
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
            timer_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="utf-8">
            <style>
              body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
              }}
              .timer-box {{
                background: #0F172A;
                border: 2px solid {'#F87171' if remaining_sec < 120 else '#818CF8'};
                border-radius: 16px;
                padding: 10px;
                text-align: center;
                box-shadow: 0 4px 14px rgba(0,0,0,0.5);
                transition: border-color 0.3s ease;
              }}
              .timer-title {{
                font-size: 0.72rem;
                font-weight: 800;
                color: #94A3B8;
                letter-spacing: 0.08em;
                text-transform: uppercase;
              }}
              .timer-digits {{
                font-size: 1.8rem;
                font-weight: 800;
                color: {'#F87171' if remaining_sec < 120 else '#818CF8'};
                font-variant-numeric: tabular-nums;
                margin-top: 2px;
                transition: color 0.3s ease;
              }}
              .timer-sub {{
                font-size: 0.68rem;
                color: #64748B;
                margin-top: 2px;
              }}
            </style>
            </head>
            <body>
              <div class="timer-box" id="tcard">
                <div class="timer-title">TIME REMAINING</div>
                <div class="timer-digits" id="cd">{timer_str}</div>
                <div class="timer-sub">Strict Exam Simulation (10 Mins)</div>
              </div>
              <script>
                var sec = {remaining_sec};
                var elem = document.getElementById("cd");
                var card = document.getElementById("tcard");
                var t = setInterval(function() {{
                  sec--;
                  if (sec <= 0) {{
                    clearInterval(t);
                    if (elem) {{
                      elem.innerText = "00:00";
                      elem.style.color = "#EF4444";
                    }}
                    if (card) card.style.borderColor = "#EF4444";
                    return;
                  }}
                  var m = Math.floor(sec / 60);
                  var s = sec % 60;
                  var ms = (m < 10 ? "0" : "") + m;
                  var ss = (s < 10 ? "0" : "") + s;
                  if (elem) {{
                    elem.innerText = ms + ":" + ss;
                    if (sec < 120) {{
                      elem.style.color = "#F87171";
                      if (card) card.style.borderColor = "#F87171";
                    }}
                  }}
                }}, 1000);
              </script>
            </body>
            </html>
            """
            components.html(timer_html, height=105)

        # Auto-submit if timer reaches zero
        if is_expired:
            st.warning("⏱ Time expired! Automatically submitting your answers...")
            graded = grade_quiz_submission(questions, st.session_state.quiz_answers)
            st.session_state.quiz_submitted = True
            st.session_state.quiz_result = graded
            st.session_state.student_accuracy = graded["percentage"]
            st.rerun()

        # Quiz Questions Form
        with st.form("quiz_form"):
            for i, q in enumerate(questions):
                st.markdown(
                    f"""
                    <div style="margin-top: 18px; margin-bottom: 8px;">
                        <span style="background: rgba(99, 102, 241, 0.15); color: #A5B4FC; border: 1px solid rgba(99, 102, 241, 0.3); font-weight: 700; font-size: 0.78rem; padding: 3px 10px; border-radius: 6px;">
                            Question {q.id} of 10 • {q.type}
                        </span>
                        <div style="font-size: 1.08rem; font-weight: 700; color: #FFFFFF; margin-top: 8px; line-height: 1.5;">
                            {q.question}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Find previous answer index if any
                saved_ans = st.session_state.quiz_answers.get(q.id, None)
                def_idx = q.options.index(saved_ans) if saved_ans in q.options else None

                selected_opt = st.radio(
                    label=f"Q{q.id} Options",
                    options=q.options,
                    index=def_idx,
                    key=f"mcq_{q.id}",
                    label_visibility="collapsed",
                )
                if selected_opt:
                    st.session_state.quiz_answers[q.id] = selected_opt

            st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
            submit_quiz = st.form_submit_button("Submit Quiz →", type="primary", use_container_width=True)

            if submit_quiz:
                graded = grade_quiz_submission(questions, st.session_state.quiz_answers)
                st.session_state.quiz_submitted = True
                st.session_state.quiz_result = graded
                st.session_state.student_accuracy = graded["percentage"]
                st.rerun()


# ─── PAGE 5: BOOK EVIDENCE ───────────────────────────────────────────────────
elif st.session_state.page == "evidence":
    retrieved = st.session_state.retrieved_chunks
    if not retrieved:
        st.info("No sources retrieved yet. Start an analysis to inspect exact textbook evidence.")
        if st.button("Go to Analyze", type="primary"):
            navigate_to("analyze")
        st.stop()

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div class="hero-tag">AUTHORITATIVE SOURCES • COPYRIGHT SAFE</div>
            <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">Exact Book Evidence</h1>
            <p style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">
                Inspect verified textbook passages, syllabus outcomes, and historical paper logs.
                MediCompass prioritizes concept synthesis and only displays short excerpts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs by source type
    tab_punjab, tab_fed, tab_syl, tab_past = st.tabs([
        "📘 Punjab Book",
        "📗 Federal Book",
        "📑 Syllabus Outcomes",
        "📌 Past Paper Logs"
    ])

    def render_chunks_for_type(chunks, target_type):
        matching = [c for c in chunks if c.source_type == target_type]
        if not matching:
            st.info(f"No direct excerpts retrieved for {target_type} in this query.")
            return

        for chunk in matching:
            page_display = f"Page {chunk.page}" if chunk.page != "unknown" else "Page information unavailable"
            st.markdown(
                f"""
                <div class="fintech-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 800; font-size: 0.95rem; color: #FFFFFF;">{chunk.title}</span>
                        <span class="badge-verified">{chunk.source_type}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 10px;">
                        <b style="color: #CBD5E1;">{chunk.chapter or 'General Section'}</b> • {page_display} • Chunk ID: <code>{chunk.id}</code>
                    </div>
                    <div class="source-excerpt">
                        "{chunk.text}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_punjab:
        render_chunks_for_type(retrieved, "Punjab Book")

    with tab_fed:
        render_chunks_for_type(retrieved, "Federal Book")

    with tab_syl:
        render_chunks_for_type(retrieved, "Syllabus")

    with tab_past:
        render_chunks_for_type(retrieved, "Past Paper")

    # Bottom Actions
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    c_ev1, c_ev2 = st.columns(2)
    with c_ev1:
        if st.button("← Back to Concept Intelligence", use_container_width=True):
            navigate_to("concept")
    with c_ev2:
        if st.button("Take 10-MCQ Concept Check →", type="primary", use_container_width=True):
            navigate_to("quiz")
