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

    # Try loading cached index first, self-heal if missing or empty
    if not vs.load_cached_index(fp) or len(vs.chunks) == 0:
        chunks = build_knowledge_base_chunks(
            sources_dir=SOURCES_DIR,
            manifest_path=MANIFEST_PATH,
            past_papers_path=PAST_PAPERS_PATH,
        )
        vs.build_index(chunks, fp)

    return vs, fp


# ─── SECRETS CONFIGURATION ───────────────────────────────────────────────────
drive_urls_raw = st.secrets.get("GOOGLE_DRIVE_FOLDER_URLS", os.environ.get("GOOGLE_DRIVE_FOLDER_URLS", ""))
has_secret_urls = bool(
    drive_urls_raw
    and drive_urls_raw.strip()
    and any(not l.strip().startswith("#") for l in drive_urls_raw.splitlines() if l.strip())
)

# Retrieve vector store immediately on startup (zero blocking)
vector_store, current_fingerprint = get_initialized_vector_store()

# Initialize Groq Client & Session API Key
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

groq_model = st.secrets.get("GROQ_MODEL", os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
groq_client = GroqClient(api_key=st.session_state.groq_api_key, model=groq_model)
quiz_gen = QuizGenerator(api_key=st.session_state.groq_api_key, model=groq_model)


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
    st.markdown("<div class='sidebar-section-title'>CURRENT TARGET</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background-color: #0F172A; border: 1.5px solid #1E293B; border-radius: 12px; padding: 12px; margin-bottom: 16px;">
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">Subject</div>
            <div style="font-size: 0.95rem; font-weight: 800; color: #FFFFFF; margin-bottom: 6px;">{st.session_state.subject}</div>
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">Target Exam</div>
            <div style="font-size: 0.95rem; font-weight: 800; color: #818CF8;">{st.session_state.exam}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # System Status
    st.markdown("<div class='sidebar-section-title'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    chunk_count = len(vector_store.chunks)
    has_api_key = bool(st.session_state.groq_api_key and not st.session_state.groq_api_key.startswith("gsk_your_groq_api_key"))

    st.markdown(
        f"""
        <div style="background-color: #0F172A; border: 1.5px solid #1E293B; border-radius: 12px; padding: 12px; margin-bottom: 16px; font-size: 0.8rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #94A3B8;">• Knowledge Chunks:</span>
                <span style="color: #10B981; font-weight: 700;">{chunk_count} verified</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #94A3B8;">• Model:</span>
                <span style="color: #F8FAFC; font-weight: 600;">{groq_model}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94A3B8;">• Mode:</span>
                <span style="color: {'#10B981' if has_api_key else '#F59E0B'}; font-weight: 700;">
                    {'Live Groq API' if has_api_key else 'Verified Grounded Demo'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Expandable Settings (Groq API Key)
    with st.expander("⚙️ API Configuration", expanded=not has_api_key):
        user_key = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_...",
            help="Free Groq API key available at console.groq.com",
            key="groq_key_input",
        )
        if user_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = user_key
            groq_client.api_key = user_key
            quiz_gen.api_key = user_key
            st.rerun()

        st.caption("Get your free API key at [console.groq.com](https://console.groq.com)")

    # Expandable Knowledge Base Sync
    with st.expander("📁 Knowledge Base & Sync", expanded=False):
        st.markdown(
            "<div style='font-size: 0.78rem; color: #94A3B8; margin-bottom: 8px; line-height: 1.4; font-weight: 600;'>"
            "Option 1: Direct PDF Upload"
            "</div>",
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload Official Book PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="pdf_uploader",
        )
        if uploaded_files:
            os.makedirs(SOURCES_DIR, exist_ok=True)
            saved_count = 0
            for upf in uploaded_files:
                dest_path = os.path.join(SOURCES_DIR, upf.name)
                if not os.path.exists(dest_path):
                    with open(dest_path, "wb") as f:
                        f.write(upf.getbuffer())
                    saved_count += 1
            if saved_count > 0:
                st.cache_resource.clear()
                st.success(f"Added {saved_count} new PDF(s)! Updating index...")
                time.sleep(1.0)
                st.rerun()

        st.markdown(
            "<div style='font-size: 0.78rem; color: #94A3B8; margin-top: 12px; margin-bottom: 6px; line-height: 1.4; font-weight: 600;'>"
            "Option 2: Sync Public Google Drive Links"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size: 0.74rem; color: #64748B; margin-bottom: 6px; line-height: 1.3;'>"
            "Must be set to <b>Anyone with the link (Viewer)</b>."
            "</div>",
            unsafe_allow_html=True,
        )
        user_drive_input = st.text_area(
            "Drive Links",
            value="" if not has_secret_urls else drive_urls_raw.strip(),
            placeholder="https://drive.google.com/drive/folders/YOUR_FOLDER_ID\nhttps://drive.google.com/file/d/YOUR_FILE_ID/view",
            height=70,
            label_visibility="collapsed",
            key="gdrive_input_box",
        )
        if st.button("📥 Sync Google Drive", key="btn_sync_gdrive", type="primary", use_container_width=True):
            target_links = [
                u.strip()
                for u in user_drive_input.splitlines()
                if u.strip() and not u.strip().startswith("#")
            ]
            if not target_links:
                st.warning("Please enter at least one valid Google Drive link.")
            else:
                with st.spinner("Downloading documents from Google Drive..."):
                    downloaded = sync_google_drive_public_folders(target_links, SOURCES_DIR)
                    if downloaded:
                        st.cache_resource.clear()
                        st.success(f"Downloaded {len(downloaded)} file(s)! Updating RAG index...")
                        time.sleep(1.2)
                        st.rerun()
                    else:
                        st.error("No files downloaded. Check that link is set to 'Anyone with the link can view' or use Direct Upload above.")

    if st.button("⚡ Reload Knowledge Base", key="reload_kb", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


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

    if len(vector_store.chunks) == 0:
        st.warning(
            """
            **Knowledge base not configured.**
            Add authorized PDFs to `data/sources/` or configure `GOOGLE_DRIVE_FOLDER_URLS` in Streamlit Secrets.
            """
        )
    elif has_secret_urls and len(vector_store.chunks) <= 12:
        st.info("💡 **Google Drive Study Sources Connected**: Your Google Drive folder is configured. Expand **Knowledge Base & Sync** in the left sidebar and click **📥 Sync Google Drive** to download and index your full textbook library into the live RAG engine.")

    # Selection Row
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

    # Input Box
    query_input = st.text_area(
        "Enter Exam Question, Topic, or Concept",
        value=st.session_state.query,
        placeholder="Example:\nWhich type of enzyme inhibition increases Km without changing Vmax?\n\nOr:\nEnzyme inhibition",
        height=130,
        help="You can enter a full multiple-choice question stem, a specific topic, or a short keyword.",
    )

    # Quick Example Chips
    st.markdown("<div style='font-size: 0.85rem; color: #CBD5E1; margin-bottom: 8px; font-weight: 700;'>Quick test prompts:</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("📌 Enzyme Inhibition & Kinetics", use_container_width=True):
        query_input = "Which type of enzyme inhibition increases Km without changing Vmax?"
        st.session_state.query = query_input
        st.rerun()
    if c2.button("📌 Overcoming Inhibition", use_container_width=True):
        query_input = "How does increasing substrate concentration affect competitive inhibition?"
        st.session_state.query = query_input
        st.rerun()
    if c3.button("📌 Non-Competitive vs Allosteric", use_container_width=True):
        query_input = "Why does Vmax decrease in non-competitive inhibition?"
        st.session_state.query = query_input
        st.rerun()

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Analyze Button
    if st.button("✦ Generate Study Decision", type="primary", use_container_width=True):
        if not query_input or not query_input.strip():
            st.error("Please enter a question or topic first.")
        else:
            st.session_state.query = query_input
            with st.spinner("Searching authoritative textbooks, PMDC syllabus, and verified past papers..."):
                retrieved_chunks = vector_store.search(
                    query=query_input,
                    subject=selected_subject,
                    exam=selected_exam,
                    top_k=8,
                )

                if not retrieved_chunks:
                    st.warning("No matching chunks found. Using foundational curriculum seed chunks.")
                    retrieved_chunks = vector_store.chunks[:5]

                analysis_result = groq_client.analyze_concept(
                    query=query_input,
                    retrieved_chunks=retrieved_chunks,
                    subject=selected_subject,
                    exam=selected_exam,
                )

                st.session_state.analysis = analysis_result
                st.session_state.quiz = None
                st.session_state.quiz_submitted = False
                navigate_to("concept")


# ─── PAGE 2: CONCEPT INTELLIGENCE ─────────────────────────────────────────────
elif st.session_state.page == "concept":
    analysis = st.session_state.analysis

    if not analysis:
        st.info("No active analysis found. Please run a study query first.")
        if st.button("← Go to Analyze"):
            navigate_to("analyze")
    else:
        col_back, col_actions = st.columns([1, 2])
        with col_back:
            if st.button("← New Question", use_container_width=True):
                navigate_to("analyze")

        with col_actions:
            a1, a2, a3 = st.columns(3)
            if a1.button("🧠 Diagram", use_container_width=True):
                navigate_to("diagram")
            if a2.button("✓ 10-MCQ Quiz", type="primary", use_container_width=True):
                navigate_to("quiz")
            if a3.button("▤ Book Evidence", use_container_width=True):
                navigate_to("evidence")

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # Main Decision Hero Card
        priority_val = analysis.get("exam_priority", "HIGH")
        ring_html = render_fintech_priority_ring(priority_val, size=115)

        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                    <div style="flex: 1; min-width: 280px;">
                        <div style="display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;">
                            <span class="badge-exam">{st.session_state.subject}</span>
                            <span class="badge-exam">{st.session_state.exam}</span>
                            <span class="badge-verified">✓ 100% Grounded in Official Sources</span>
                        </div>
                        <h2 style="font-size: 1.5rem; font-weight: 800; color: #FFFFFF; margin: 0 0 12px 0;">
                            {analysis.get('core_concept', 'Concept Intelligence')}
                        </h2>
                        <div style="background-color: #18182E; border-left: 4px solid #6366F1; padding: 14px 18px; border-radius: 0 12px 12px 0; margin-bottom: 14px;">
                            <div style="font-size: 0.74rem; font-weight: 800; color: #818CF8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                                THE DECISION FOR EXAM SUCCESS
                            </div>
                            <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; line-height: 1.5;">
                                {analysis.get('the_decision', '')}
                            </div>
                        </div>
                    </div>
                    <div>
                        {ring_html}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3 Column Intelligence Grid
        c_left, c_mid, c_right = st.columns([1, 1, 1])

        with c_left:
            st.markdown(
                f"""
                <div class="fintech-card" style="height: 100%;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px;">
                        ◈ REASONING & EXPLANATION
                    </div>
                    <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
                        {analysis.get('explanation', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_mid:
            st.markdown(
                f"""
                <div class="fintech-card" style="height: 100%; border-color: #B45309;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #F59E0B; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px;">
                        ⚠️ EXAM TRAP TO AVOID
                    </div>
                    <div style="font-size: 0.92rem; color: #FDE68A; line-height: 1.6;">
                        {analysis.get('trap', '')}
                    </div>
                    <div style="margin-top: 14px; font-size: 0.82rem; color: #FCD34D;">
                        <b>Past Frequency:</b> {analysis.get('past_paper_frequency', 'High Yield in recent cycles')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_right:
            st.markdown(
                f"""
                <div class="fintech-card" style="height: 100%;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #10B981; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px;">
                        🧠 HIGH-YIELD MEMORY HOOK
                    </div>
                    <div style="font-size: 0.98rem; font-weight: 700; color: #FFFFFF; line-height: 1.6; margin-bottom: 12px;">
                        {analysis.get('memory_hook', '')}
                    </div>
                    <div style="background-color: #022C22; border: 1px solid #059669; border-radius: 8px; padding: 10px; font-size: 0.8rem; color: #6EE7B7;">
                        <b>Pro Study Tip:</b> Memorize the kinetic graph axis changes to lock in +4 points.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── PAGE 3: QUICK DIAGRAM ────────────────────────────────────────────────────
elif st.session_state.page == "diagram":
    analysis = st.session_state.analysis

    if not analysis:
        st.info("No active analysis found. Please run a study query first.")
        if st.button("← Go to Analyze"):
            navigate_to("analyze")
    else:
        st.markdown(
            f"""
            <div style="margin-bottom: 20px;">
                <span class="badge-exam">{st.session_state.subject}</span>
                <h2 style="font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin: 8px 0;">
                    🧠 Mental Model Diagram: {analysis.get('core_concept', '')}
                </h2>
                <p style="color: #94A3B8; font-size: 0.95rem;">
                    Visual process flow structured from authoritative textbook explanations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        diagram_spec = analysis.get("diagram", {})
        if not diagram_spec:
            diagram_spec = {
                "type": "ENZYMATIC KINETIC CASCADE",
                "title": analysis.get("core_concept", "Mechanism"),
                "steps": [
                    {"title": "1. Normal State", "desc": "Substrate binds catalytic site at steady state (Vmax)."},
                    {"title": "2. Inhibitor Challenge", "desc": "Competitive inhibitor competes for same active site."},
                    {"title": "3. Substrate Shift", "desc": "Increasing [S] outcompetes inhibitor, restoring Vmax (Km increases)."},
                ],
            }

        st.markdown(render_concept_diagram_html(diagram_spec), unsafe_allow_html=True)

        st.markdown(
            """
            <div style="margin-top: 24px;">
                <h3 style="font-size: 1.1rem; font-weight: 800; color: #FFFFFF;">ASCII Logic Diagram</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ascii_diagram = """
┌───────────────────────────┐      High [Substrate]      ┌───────────────────────────┐
│   Competitive Inhibitor   │ ─────────────────────────> │   Inhibition Overcome     │
│  Competes for Active Site │                            │    Vmax Unchanged         │
└───────────────────────────┘                            └───────────────────────────┘
              │                                                        ▲
              ▼                                                        │
┌───────────────────────────┐                            ┌───────────────────────────┐
│     Apparent Km Rises     │ ─────────────────────────> │   Lower Apparent Affinity │
└───────────────────────────┘                            └───────────────────────────┘
        """
        st.code(ascii_diagram, language="text")

        c_back, c_quiz = st.columns([1, 1])
        with c_back:
            if st.button("← Back to Concept Intelligence", use_container_width=True):
                navigate_to("concept")
        with c_quiz:
            if st.button("Take 10-MCQ Quiz →", type="primary", use_container_width=True):
                navigate_to("quiz")


# ─── PAGE 4: 10-MCQ QUIZ ──────────────────────────────────────────────────────
elif st.session_state.page == "quiz":
    analysis = st.session_state.analysis

    if not analysis:
        st.info("No active analysis found. Please run a study query first.")
        if st.button("← Go to Analyze"):
            navigate_to("analyze")
    else:
        # Load or generate quiz
        if st.session_state.quiz is None:
            with st.spinner("Generating targeted 10-MCQ Concept Check from verified past papers..."):
                retrieved_chunks = vector_store.search(
                    query=analysis.get("core_concept", st.session_state.query),
                    subject=st.session_state.subject,
                    exam=st.session_state.exam,
                    top_k=6,
                )
                generated_quiz = quiz_gen.generate_quiz(
                    concept=analysis.get("core_concept", "Enzyme Inhibition"),
                    retrieved_chunks=retrieved_chunks,
                    subject=st.session_state.subject,
                    exam=st.session_state.exam,
                )
                st.session_state.quiz = generated_quiz
                reset_quiz_timer()

        quiz = st.session_state.quiz
        questions = quiz.get("questions", [])

        st.markdown(
            f"""
            <div style="margin-bottom: 24px;">
                <div class="hero-tag">MDCAT & NUMS TEST SIMULATION</div>
                <h1 class="hero-title">10-MCQ Concept Mastery Check</h1>
                <p class="hero-sub">
                    Topic: <b>{quiz.get('topic', 'Concept Check')}</b> • Target: <b>{st.session_state.exam}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.quiz_submitted:
            with st.form("quiz_form"):
                for q in questions:
                    q_id = q.get("id", 1)
                    st.markdown(
                        f"""
                        <div style="background-color: #0F172A; border: 1.5px solid #1E293B; border-radius: 16px; padding: 20px; margin-bottom: 18px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span style="background-color: #1E293B; color: #818CF8; font-weight: 800; font-size: 0.75rem; padding: 4px 10px; border-radius: 6px;">
                                    QUESTION {q_id} OF 10
                                </span>
                                <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">
                                    {q.get('type', 'Conceptual Reasoning')}
                                </span>
                            </div>
                            <div style="font-size: 1.02rem; font-weight: 700; color: #FFFFFF; line-height: 1.5; margin-bottom: 14px;">
                                {q.get('question', '')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    opts = q.get("options", [])
                    st.radio(
                        f"Select your answer for Q{q_id}:",
                        opts,
                        key=f"user_q_{q_id}",
                        index=None,
                        label_visibility="collapsed",
                    )
                    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

                submit_btn = st.form_submit_button("Submit 10-Question Quiz & Grade", type="primary", use_container_width=True)
                if submit_btn:
                    answers = {}
                    for q in questions:
                        qid = q.get("id")
                        answers[qid] = st.session_state.get(f"user_q_{qid}", None)

                    score = grade_quiz_submission(quiz, answers)
                    st.session_state.quiz_submitted = True
                    st.session_state.quiz_score = score
                    st.session_state.user_answers = answers
                    st.session_state.quiz_time_taken = get_quiz_timer_state()
                    st.rerun()

        else:
            # Scoreboard View
            score = st.session_state.quiz_score
            time_taken = st.session_state.quiz_time_taken
            score_ring = render_fintech_score_ring(score, total=10, size=135)

            st.markdown(
                f"""
                <div class="fintech-card" style="text-align: center; padding: 32px 24px;">
                    <div style="display: flex; justify-content: center; margin-bottom: 16px;">
                        {score_ring}
                    </div>
                    <h2 style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF; margin: 0 0 8px 0;">
                        {'Outstanding Mastery! 🎯' if score >= 8 else ('Good Effort! Review High Yield Traps ⚠️' if score >= 5 else 'Needs Dedicated Concept Review 📚')}
                    </h2>
                    <p style="color: #94A3B8; font-size: 0.95rem; margin: 0 0 20px 0;">
                        Time elapsed: <b>{time_taken // 60}m {time_taken % 60}s</b> • Accuracy: <b>{score * 10}%</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Question by question review
            st.markdown("<h3 style='font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin: 24px 0 16px 0;'>Detailed Question Breakdown & Explanations</h3>", unsafe_allow_html=True)

            for q in questions:
                q_id = q.get("id")
                correct = q.get("answer")
                user_ans = st.session_state.user_answers.get(q_id)
                is_correct = (user_ans == correct)

                badge_color = "#10B981" if is_correct else "#EF4444"
                badge_text = "✓ CORRECT (+1)" if is_correct else "✗ INCORRECT (0)"

                st.markdown(
                    f"""
                    <div class="fintech-card" style="border-color: {'#065F46' if is_correct else '#991B1B'};">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="font-size: 0.8rem; font-weight: 800; color: {badge_color};">
                                {badge_text}
                            </span>
                            <span style="font-size: 0.76rem; color: #94A3B8; font-weight: 600;">
                                Past Reference: {q.get('past_paper', 'MDCAT verified')}
                            </span>
                        </div>
                        <div style="font-size: 1.02rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px; line-height: 1.5;">
                            Q{q_id}: {q.get('question', '')}
                        </div>
                        <div style="font-size: 0.88rem; margin-bottom: 6px; color: #CBD5E1;">
                            <b>Your Answer:</b> <span style="color: {'#6EE7B7' if is_correct else '#FCA5A5'}; font-weight: 700;">{user_ans or 'Unanswered'}</span>
                        </div>
                        <div style="font-size: 0.88rem; margin-bottom: 12px; color: #CBD5E1;">
                            <b>Correct Key:</b> <span style="color: #6EE7B7; font-weight: 700;">{correct}</span>
                        </div>
                        <div style="background-color: #080B11; border: 1px solid #1E293B; border-radius: 10px; padding: 12px; font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
                            <b>Why?</b> {q.get('explanation', '')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.button("↺ Retake Quiz / Generate New Variants", type="primary", use_container_width=True):
                st.session_state.quiz = None
                st.session_state.quiz_submitted = False
                st.rerun()


# ─── PAGE 5: BOOK EVIDENCE ────────────────────────────────────────────────────
elif st.session_state.page == "evidence":
    analysis = st.session_state.analysis

    if not analysis:
        st.info("No active analysis found. Please run a study query first.")
        if st.button("← Go to Analyze"):
            navigate_to("analyze")
    else:
        st.markdown(
            f"""
            <div style="margin-bottom: 24px;">
                <div class="hero-tag">PRIMARY SOURCE CROSS-REFERENCE</div>
                <h1 class="hero-title">Authoritative Evidence Citations</h1>
                <p class="hero-sub">
                    Direct citations from official provincial textbooks and PMDC curriculum.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        evidence = analysis.get("evidence", {})

        # Punjab Textbook Board
        ptb = evidence.get("punjab_textbook", {})
        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">🏛</span>
                        <h3 style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin: 0;">Punjab Curriculum & Textbook Board (PTB)</h3>
                    </div>
                    <span class="badge-exam">Page {ptb.get('page', '3')}</span>
                </div>
                <div style="font-size: 0.85rem; color: #818CF8; font-weight: 700; margin-bottom: 8px;">
                    {ptb.get('chapter', 'Chapter 11: Enzymes and Metabolism')} • Edition: {ptb.get('edition', '2023-24')}
                </div>
                <div class="source-excerpt">
                    "{ptb.get('excerpt', 'Competitive inhibitors resemble substrate and can be overcome by increasing substrate.')}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Federal Board / National Book Foundation
        fed = evidence.get("federal_textbook", {})
        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">📘</span>
                        <h3 style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin: 0;">Federal Board (National Book Foundation)</h3>
                    </div>
                    <span class="badge-exam">Page {fed.get('page', '2')}</span>
                </div>
                <div style="font-size: 0.85rem; color: #818CF8; font-weight: 700; margin-bottom: 8px;">
                    {fed.get('chapter', 'Chapter 3: Enzymes')} • Edition: {fed.get('edition', '2023-24')}
                </div>
                <div class="source-excerpt">
                    "{fed.get('excerpt', 'In competitive inhibition, Km increases because higher substrate concentration is required.')}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # PMDC Syllabus Outcome
        pmdc = evidence.get("pmdc_syllabus", {})
        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">📋</span>
                        <h3 style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin: 0;">PMDC Official Curriculum Outcome</h3>
                    </div>
                    <span class="badge-verified">Official Learning Outcome</span>
                </div>
                <div style="font-size: 0.85rem; color: #818CF8; font-weight: 700; margin-bottom: 8px;">
                    Standard: {pmdc.get('standard', '2.4')}
                </div>
                <div class="source-excerpt">
                    "{pmdc.get('outcome', 'Distinguish between competitive and non-competitive inhibitors in terms of binding site, substrate competition, Km and Vmax.')}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("← Back to Concept Intelligence", use_container_width=True):
            navigate_to("concept")


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; margin-top: 40px; padding: 20px 0; border-top: 1px solid #1E293B; color: #64748B; font-size: 0.78rem;">
        MediCompass AI • Authoritative Medical Entrance Test Intelligence • Grounded in Punjab & Federal Textbooks
    </div>
    """,
    unsafe_allow_html=True,
)
