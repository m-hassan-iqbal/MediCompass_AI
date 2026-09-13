"""
MediCompass AI - Master Streamlit Application
Turning exam information into study intelligence for MDCAT and NUMS aspirants.
"""

import os
import time
import streamlit as st

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


# Retrieve vector store
vector_store, current_fingerprint = get_initialized_vector_store()

# Initialize Groq Client
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
groq_model = st.secrets.get("GROQ_MODEL", os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
groq_client = GroqClient(api_key=groq_api_key, model=groq_model)
quiz_gen = QuizGenerator(api_key=groq_api_key, model=groq_model)


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
    chunk_count = len(vector_store.chunks)
    groq_ready = groq_client.is_configured()

    st.markdown("<div style='font-size: 0.78rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.7; background: #0F172A; border: 1.5px solid #1E293B; border-radius: 12px; padding: 12px 14px;">
            <div>• Knowledge Chunks: <b style="color: #FFFFFF;">{chunk_count} verified</b></div>
            <div>• Model: <b style="color: #FFFFFF;">{groq_model}</b></div>
            <div>• Mode: <span style="color: {'#4ADE80' if groq_ready else '#FBBF24'}; font-weight: 800;">{'Live Groq API' if groq_ready else 'Verified Seed Demo'}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not groq_ready:
        st.caption("Add GROQ_API_KEY to Streamlit Secrets to enable arbitrary query generation.")

    if chunk_count == 0 or st.button("⚡ Reload Knowledge Base", key="reload_kb", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    # Google Drive Sync Trigger (via Secrets or Direct UI Paste)
    drive_urls_raw = st.secrets.get("GOOGLE_DRIVE_FOLDER_URLS", "")
    has_secret_urls = bool(
        drive_urls_raw
        and drive_urls_raw.strip()
        and any(not l.strip().startswith("#") for l in drive_urls_raw.splitlines() if l.strip())
    )

    with st.expander("🔗 Connect Google Drive Sources", expanded=False):
        st.markdown(
            "<div style='font-size: 0.8rem; color: #94A3B8; margin-bottom: 8px; line-height: 1.4;'>"
            "Paste your public Google Drive folder or PDF links below.<br/>"
            "<span style='color: #818CF8;'>Note:</span> Sharing must be set to <b>Anyone with the link (Viewer)</b>."
            "</div>",
            unsafe_allow_html=True,
        )
        user_drive_input = st.text_area(
            "Drive Links",
            value="" if not has_secret_urls else drive_urls_raw.strip(),
            placeholder="https://drive.google.com/drive/folders/YOUR_FOLDER_ID\nhttps://drive.google.com/file/d/YOUR_FILE_ID/view",
            height=90,
            label_visibility="collapsed",
            key="gdrive_input_box",
        )
        if st.button("📥 Sync & Ingest Drive Sources", key="btn_sync_gdrive", type="primary", use_container_width=True):
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
                        st.error("Download failed or no files found. Make sure General Access is set to 'Anyone with the link' (Viewer).")


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

    # If Knowledge base has no sources
    if len(vector_store.chunks) == 0:
        st.warning(
            """
            **Knowledge base not configured.**
            Add authorized PDFs to `data/sources/` or configure `GOOGLE_DRIVE_FOLDER_URLS` in Streamlit Secrets.
            """
        )

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

            # RAG Retrieval
            retrieved = vector_store.retrieve_relevant_chunks(
                query=query_input,
                subject=st.session_state.subject,
                exam=st.session_state.exam,
                top_k=6,
            )
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

    # Header section
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

    # 3 Primary Action Buttons
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

    # Top Grid: Core Concept Card + Study Priority Ring
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

        # Dynamic formula explanation based on whether student has attempted the quiz
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
                    UNDERSTAND THE CONCEPT
                </span>
                <h3 style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 14px 0;">
                    Mechanistic Breakdown
                </h3>
                <ul style="padding-left: 20px; margin: 0;">
                    {pts_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Source Synthesis: Punjab + Federal
    st.markdown(
        f"""
        <div class="fintech-card">
            <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                SOURCE SYNTHESIS
            </span>
            <h3 style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin: 4px 0 16px 0;">
                Punjab vs. Federal Textbook Coverage
            </h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div style="background: #080B11; border: 1px solid #1E293B; border-radius: 14px; padding: 16px;">
                    <div style="font-weight: 800; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 6px;">
                        📘 Punjab Textbook Board
                    </div>
                    <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5;">
                        {analysis.get('punjab_synthesis', 'Relevant concept found in Punjab textbook.')}
                    </div>
                </div>
                <div style="background: #080B11; border: 1px solid #1E293B; border-radius: 14px; padding: 16px;">
                    <div style="font-weight: 800; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 6px;">
                        📗 Federal Board / NBF
                    </div>
                    <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5;">
                        {analysis.get('federal_synthesis', 'Supporting concept found in Federal textbook.')}
                    </div>
                </div>
            </div>
            <div style="margin-top: 14px; padding: 12px 16px; background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; font-size: 0.9rem; color: #CBD5E1;">
                <b style="color: #A5B4FC;">MediCompass Takeaway:</b> {analysis.get('synthesis_takeaway', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Historical Past-Paper Intelligence & Syllabus Alignment
    col_past, col_syl = st.columns([1.5, 1])

    with col_past:
        past_items = analysis.get("past_paper_evidence", [])
        past_html = ""
        if past_items:
            for p in past_items:
                year = p.get("year", "Historical")
                exam_name = p.get("exam", "MDCAT")
                summary = p.get("summary", "")
                past_html += f"""
                <div style="margin-bottom: 10px; padding: 10px 14px; background: #080B11; border-radius: 10px; border: 1px solid #1E293B; border-left: 3px solid #6366F1;">
                    <div style="font-weight: 700; font-size: 0.85rem; color: #FFFFFF;">📌 {year} · {exam_name}</div>
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 2px;">{summary}</div>
                </div>
                """
        else:
            past_html = "<div style='font-size: 0.85rem; color: #94A3B8;'>No direct past paper occurrences identified in current verified logs.</div>"

        st.markdown(
            f"""
            <div class="fintech-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.75rem; font-weight: 800; color: #818CF8; letter-spacing: 0.08em; text-transform: uppercase;">
                        HISTORICAL EXAM EVIDENCE
                    </span>
                    <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">Historical Evidence ≠ Prediction</span>
                </div>
                {past_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_syl:
        syl_status = analysis.get("syllabus_status", "Covered")
        syl_details = analysis.get("syllabus_details", "")
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
        st.markdown(
            """
            <div style="margin-bottom: 20px;">
                <div class="hero-tag">EVALUATION • DIAGNOSTIC SUMMARY</div>
                <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0;">10-MCQ Concept Check Results</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_score, col_feedback = st.columns([1.2, 2.4])
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

        # Question Review Section
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

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    if st.button("← Return to Concept Intelligence", type="primary", use_container_width=True):
        navigate_to("concept")


# ─── RESPONSIBLE EDUCATIONAL FOOTER ──────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top: 50px; padding: 24px 0; border-top: 1.5px solid #1E293B; text-align: center; font-size: 0.85rem; color: #94A3B8; line-height: 1.6; font-weight: 500;">
        <b style="color: #FFFFFF;">MediCompass AI</b> — Turning exam information into study intelligence.<br/>
        Evidence available ≠ question prediction. MediCompass is an educational study intelligence system.<br/>
        Always verify important curriculum materials against official PMDC syllabus and authorized textbooks.
    </div>
    """,
    unsafe_allow_html=True,
)
