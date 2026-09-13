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
    chunk_count = len(getattr(vector_store, "chunks", []))
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

    if len(getattr(vector_store, "chunks", [])) == 0:
        st.warning(
            """
            **Knowledge base not configured.**
            Add authorized PDFs to `data/sources/` or configure `GOOGLE_DRIVE_FOLDER_URLS` in Streamlit Secrets.
            """
        )
    elif has_secret_urls and len(getattr(vector_store, "chunks", [])) <= 12:
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
    if c3.button("📌 Allosteric vs Active Site", use_container_width=True):
        query_input = "Differences between competitive and non-competitive enzyme inhibition"
        st.session_state.query = query_input
        st.rerun()

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # CTA Button
    if st.button("Analyze with MediCompass →", type="primary", use_container_width=True):
        if not query_input.strip():
            st.error("Please enter a question or topic to analyze.")
        else:
            st.session_state.query = query_input

            # Multi-stage progress experience
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
                time.sleep(0.3)

            # RAG Retrieval (defensive call)
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

            # Groq Analysis
            analysis = groq_client.analyze_concept(
                query=query_input,
                subject=st.session_state.subject,
                exam=st.session_state.exam,
                retrieved_chunks=retrieved,
            )
            st.session_state.analysis = analysis

            # Pre-generate 10-MCQ Quiz
            quiz_qs = quiz_gen.generate_10_mcq_quiz(
                concept_title=analysis.get("concept_title", "Enzyme Kinetics"),
                subject=st.session_state.subject,
                exam=st.session_state.exam,
                retrieved_chunks=retrieved,
            )
            st.session_state.quiz_questions = quiz_qs
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = None
            st.session_state.quiz_time_taken = 0
            reset_quiz_timer()

            progress_placeholder.empty()
            progress_bar.empty()
            navigate_to("concept")


# ─── PAGE 2: CONCEPT INTELLIGENCE ─────────────────────────────────────────────
elif st.session_state.page == "concept":
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Please enter a question or topic on the Analyze page first.")
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
                <div>
                    <span class="badge-exam">{st.session_state.subject}</span>
                    <span class="badge-exam">{st.session_state.exam}</span>
                    <span class="badge-verified">✓ {len(analysis.get('source_ids', []))} Verified Citations</span>
                    <h1 class="hero-title" style="margin-top: 10px; font-size: 2rem;">{analysis.get('concept_title', 'Concept Analysis')}</h1>
                    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: -6px;">{analysis.get('question_summary', '')}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown(
                f"""
                <div class="fintech-card">
                    <div style="font-size: 0.78rem; font-weight: 800; color: #818CF8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
                        THE STUDY DECISION
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; line-height: 1.6;">
                        {analysis.get('core_explanation', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='font-weight: 800; font-size: 1.1rem; color: #FFFFFF; margin: 24px 0 12px 0;'>Authoritative Breakdown</div>", unsafe_allow_html=True)
            for item in analysis.get("deep_explanation", []):
                st.markdown(
                    f"""
                    <div style="background: #0F172A; border-left: 3px solid #6366F1; border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 10px; font-size: 0.92rem; color: #E2E8F0; line-height: 1.5;">
                        {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_right:
            priority_score = analysis.get("priority_score", 70)
            priority_label = analysis.get("priority_label", "STUDY NOW")
            ring_html = render_fintech_priority_ring(priority_label, size=130)

            st.markdown(
                f"""
                <div class="fintech-card" style="text-align: center;">
                    {ring_html}
                    <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; margin-top: 10px;">
                        Priority: {priority_label}
                    </div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 8px; line-height: 1.4;">
                        {analysis.get('priority_reason', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            hook = analysis.get("memory_hook", "")
            if hook:
                st.markdown(
                    f"""
                    <div class="memory-box">
                        <div style="font-size: 0.72rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px;">
                            💡 MEMORY HOOK
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 700; line-height: 1.4;">
                            {hook}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ─── PAGE 3: QUICK DIAGRAM ────────────────────────────────────────────────────
elif st.session_state.page == "diagram":
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Run an analysis first to view its concept diagram.")
    else:
        diagram_spec = analysis.get("diagram", {})
        st.markdown("<h2 class='hero-title' style='font-size: 1.8rem;'>🧠 Concept Diagram & Mental Flow</h2>", unsafe_allow_html=True)
        st.markdown(render_concept_diagram_html(diagram_spec), unsafe_allow_html=True)


# ─── PAGE 4: 10-MCQ QUIZ ──────────────────────────────────────────────────────
elif st.session_state.page == "quiz":
    analysis = st.session_state.analysis
    quiz_qs = st.session_state.get("quiz_questions", [])

    if not analysis or not quiz_qs:
        st.info("Run an analysis first to generate your 10-MCQ test.")
    else:
        st.markdown("<h2 class='hero-title' style='font-size: 1.8rem;'>✓ 10-MCQ Concept Mastery Test</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; margin-top: -6px;'>Directly calibrated to MDCAT & NUMS past paper patterns.</p>", unsafe_allow_html=True)

        with st.form("quiz_form"):
            for idx, q in enumerate(quiz_qs):
                st.markdown(
                    f"""
                    <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 14px; padding: 18px; margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8;">QUESTION {idx + 1} OF 10</span>
                            <span style="font-size: 0.75rem; color: #94A3B8;">{q.get('type', 'Conceptual Reasoning')}</span>
                        </div>
                        <div style="font-size: 1rem; font-weight: 700; color: #FFFFFF; line-height: 1.5;">
                            {q.get('question', '')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                selected_opt = st.radio(
                    f"Options for Q{idx+1}",
                    options=q.get("options", []),
                    key=f"q_{idx}",
                    index=None,
                    label_visibility="collapsed",
                )
                st.session_state.quiz_answers[idx] = selected_opt

            submit_quiz = st.form_submit_button("Submit 10-Question Test", type="primary", use_container_width=True)

        if submit_quiz:
            score, results = grade_quiz_submission(quiz_qs, st.session_state.quiz_answers)
            st.session_state.quiz_submitted = True
            st.session_state.quiz_score = score
            st.session_state.quiz_results = results
            st.session_state.quiz_time_taken = get_quiz_timer_state()
            st.rerun()

        if st.session_state.quiz_submitted:
            score = st.session_state.quiz_score
            score_ring = render_fintech_score_ring(score, total=10, size=135)

            st.markdown(
                f"""
                <div class="fintech-card" style="text-align: center; margin-top: 24px;">
                    {score_ring}
                    <h3 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin-top: 14px;">
                        {'Mastery Achieved! 🎯' if score >= 8 else ('Target Met — Review Mistakes ⚠️' if score >= 5 else 'Needs Immediate Concept Revision 📚')}
                    </h3>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── PAGE 5: BOOK EVIDENCE ────────────────────────────────────────────────────
elif st.session_state.page == "evidence":
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Run an analysis first to view its source citations.")
    else:
        st.markdown("<h2 class='hero-title' style='font-size: 1.8rem;'>▤ Authoritative Book Evidence</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; margin-top: -6px;'>Exact cross-references from official textbooks and PMDC syllabus.</p>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="font-size: 0.8rem; font-weight: 800; color: #818CF8; text-transform: uppercase; margin-bottom: 8px;">
                    🏛 Punjab Curriculum & Textbook Board (PTB)
                </div>
                <div style="font-size: 0.95rem; color: #E2E8F0; line-height: 1.6;">
                    {analysis.get('punjab_synthesis', 'Grounded in prescribed Punjab textbooks.')}
                </div>
            </div>
            <div class="fintech-card">
                <div style="font-size: 0.8rem; font-weight: 800; color: #818CF8; text-transform: uppercase; margin-bottom: 8px;">
                    📘 Federal Board / National Book Foundation
                </div>
                <div style="font-size: 0.95rem; color: #E2E8F0; line-height: 1.6;">
                    {analysis.get('federal_synthesis', 'Grounded in prescribed Federal textbooks.')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
