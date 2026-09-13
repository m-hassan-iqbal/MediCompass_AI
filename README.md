# MediCompass AI 🧭

> **Turning exam information into study intelligence.**
> Evidence-based study intelligence system for Pakistani medical entrance test students preparing for **MDCAT** and **NUMS**.

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-API%20Llama%203.3%2070B-F55036.svg)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Product Philosophy

MediCompass AI is **NOT**:
* ❌ A generic AI chatbot
* ❌ A ChatGPT clone
* ❌ A giant indiscriminate MCQ bank
* ❌ A pirated textbook repository
* ❌ A speculative question prediction engine

MediCompass AI **IS**:
> **An evidence-based study intelligence system that takes one exam question or topic and converts it into a clear concept, source evidence, historical exam relevance, study priority, memorization aid, and focused practice.**

The student's fundamental problem is **not a lack of information**. The student's problem is:
> *"I have too much information. What actually matters?"*

### Core Principle
> **Do not make students read more. Help them understand what matters.**

### Product North Star
> **"Given what I am preparing for, what do I need to understand next, which source should I use, and what evidence supports that decision?"**

---

## 2. System Architecture

MediCompass AI is architected with a strict **grounded-first** hierarchy:
```
SOURCE DOCUMENTS (Punjab, Federal, PMDC Syllabus, Past Papers)
      ↓
PDF EXTRACTION (pypdf with page preservation & unknown fallback)
      ↓
TEXT CLEANING & SEMANTIC CHUNKING (1000-1500 chars, 150-250 overlap)
      ↓
METADATA ATTACHMENT (source_type, subject, exam, chapter, page, year)
      ↓
LOCAL EMBEDDINGS (sentence-transformers/all-MiniLM-L6-v2)
      ↓
VECTOR SIMILARITY SEARCH (Cosine similarity with source-type diversity reranking)
      ↓
TOP-K DIVERSE RETRIEVAL (Syllabus + Punjab + Federal + Past Papers)
      ↓
GROQ LLM (llama-3.3-70b-versatile with JSON mode & repair fallback)
      ↓
STRUCTURED CONCEPT INTELLIGENCE
 ┌───────────────────────────┼───────────────────────────┐
 ↓                           ↓                           ↓
🧠 QUICK MEMORIZE DIAGRAM   🎯 10-MCQ CONCEPT CHECK     📖 VISIT EXACT BOOK EVIDENCE
```

---

## 3. Core Features

### ◈ Concept Intelligence
* **Core Concept Card**: Answers what it is, how it works, why it is relevant, what causes the correct result, and common student confusions.
* **Deep Mechanistic Explanation**: Bullet points labeled with visual `[VERIFIED]` (source grounded) and `[INFERENCE]` (pedagogical guidance) tags.
* **Why This Matters**: Syllabus relevance and historical exam grounding without speculative probability claims.
* **Memory Hook**: Quick mnemonics and kinetic rules for rapid recall.
* **Source Synthesis**: Direct comparison of Punjab Textbook vs. Federal Board coverage.
* **Historical Past-Paper Intelligence**: Concept logs from verified past MDCAT/NUMS papers (📌 verified occurrences only).
* **Evidence-Based Study Priority Ring**: Transparent 0–100 circular visual score:
  $$\text{Priority} = \text{Syllabus Relevance (0-40)} + \text{Historical Evidence (0-35)} + \text{Personal Performance (0-25)}$$
  * `75-100`: **STUDY NOW**
  * `50-74`: **REVIEW SOON**
  * `0-49`: **LOWER PRIORITY**

### 🧠 Quick Memorize Diagram
* Responsive, high-aesthetic educational flowchart (4–8 nodes).
* Cause-and-effect connectors and key terms.
* Rapid recall anchors for quick revision before exam day.

### 🎯 10-MCQ Concept Check
* Exactly **10 conceptual MCQs** (3 Direct Understanding, 4 Conceptual Reasoning, 3 Application / Interpretation).
* Real **10-minute countdown timer** (`MM:SS`) with automatic submission at `00:00`.
* Circular Concept Score Ring (`X / 10`).
* In-depth answer breakdown:
  * User choice vs. correct answer
  * Deep scientific concept explanation
  * Misconception analysis explaining why wrong options are tempting
  * Memory hooks and verified historical paper references

### 📖 Exact Book Evidence
* Tabbed inspection of short, verbatim source excerpts from **Punjab Books**, **Federal Books**, **Syllabus Outcomes**, and **Past Papers**.
* Exact page citations (or explicit *"Page information unavailable"* if not determinable).
* **Copyright-Safe**: Never redistributes full books or contiguous chapters.

---

## 4. Local Installation & Setup

### Prerequisites
* Python 3.11, 3.12, or 3.13
* Git

### Step-by-Step Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MediCompass_AI.git
   cd MediCompass_AI
   ```

2. **Create and activate a virtual environment**:
   * On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * On macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate seed demo sources (optional if already generated)**:
   ```bash
   python data/generate_seed_pdfs.py
   ```

5. **Configure Secrets**:
   Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Add your free Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys):
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```
   *(Note: If you run the app without setting a Groq API key, MediCompass AI automatically runs in **Verified Seed Demo Mode**, allowing you to test all 5 pages seamlessly using authentic MDCAT/NUMS Enzyme Kinetics data!)*

6. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 5. Google Drive Integration (MVP)

MediCompass AI supports syncing authorized, publicly accessible Google Drive folders.

1. In your `.streamlit/secrets.toml` or Streamlit Cloud Secrets, configure:
   ```toml
   GOOGLE_DRIVE_FOLDER_URLS = """
   https://drive.google.com/drive/folders/YOUR_PUBLIC_BIOLOGY_FOLDER_ID
   https://drive.google.com/drive/folders/YOUR_PUBLIC_CHEMISTRY_FOLDER_ID
   https://drive.google.com/drive/folders/YOUR_PUBLIC_PHYSICS_FOLDER_ID
   """
   ```
2. When configured, a **🔄 Sync Google Drive Sources** button appears in the sidebar.
3. Downloaded files are fingerprinted and cached locally in `data/sources/` to avoid redundant downloads across reruns.

> [!NOTE]
> **Private Google Drive Access**: `gdown` only supports public/shared Drive folders. For restricted enterprise or institution drives, the architecture is ready for Google Drive API v3 using Service Accounts or OAuth2 tokens. Never commit service account keys to GitHub!

---

## 6. Streamlit Community Cloud Deployment

Deploying MediCompass AI to Streamlit Cloud takes under 3 minutes:

1. Push your repository to **GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: initial MediCompass AI release"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/MediCompass_AI.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app** and select:
   * **Repository**: `YOUR_USERNAME/MediCompass_AI`
   * **Branch**: `main`
   * **Main file path**: `app.py`
4. Click **Advanced settings...** and paste your secrets:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```
5. Click **Deploy!**

---

## 7. Knowledge Base Structure

```
data/
├── source_manifest.json          # Authoritative mapping of PDFs to boards & subjects
├── past_papers.json              # Curated historical concept occurrences
└── sources/                      # Local PDF documents
    ├── Punjab_Biology_Enzymes_Ch11.pdf
    ├── Federal_Biology_Enzymes_Ch3.pdf
    └── PMDC_MDCAT_NUMS_Syllabus_Biology.pdf
```

### Manifest Schema (`source_manifest.json`)
```json
[
  {
    "file": "Punjab_Biology_Enzymes_Ch11.pdf",
    "source_type": "Punjab Book",
    "subject": "Biology",
    "exam": ["MDCAT", "NUMS"],
    "chapter": "Chapter 11: Enzymes",
    "edition": "Punjab Curriculum and Textbook Board (PTB) 2023-24",
    "authorized": true
  }
]
```

### Deterministic Fingerprinting
The local vector store hashes file names, sizes, and timestamps. If no source files have been added or edited, the embedded vector index is loaded instantly from `cache/` without recalculation.

---

## 8. Anti-Hallucination & Evidence Grounding

MediCompass AI strictly enforces three foundational safety protocols:

1. **Grounded First, Generated Second**:
   All outputs must derive from retrieved chunks. If an item is not in the knowledge base, the application explicitly outputs:
   `"Evidence not available in the current knowledge base."`
2. **No Speculative Predictions**:
   The engine **never** claims: *"This has an 80% chance of coming on the MDCAT."*
   It strictly states: *"This concept has demonstrated historical exam relevance."*
3. **Source ID Sanitization**:
   Every `source_id` returned by the LLM is programmatically validated against the actual chunks retrieved in that session. Hallucinated IDs are discarded.

---

## 9. Automated Testing Suite

Run the full test suite with pytest:

```bash
pytest tests/
```

### Test Coverage:
* `tests/test_chunking.py`: Validates text normalization, boundary-aware semantic chunking, and PDF page extraction.
* `tests/test_quiz_validation.py`: Enforces exactly 10 questions, 4 options, valid answer matching, and grading calculation across mastery tiers.
* `tests/test_priority.py`: Asserts strict 0–100 mathematical bounds, priority threshold mapping (`STUDY NOW`, `REVIEW SOON`, `LOWER PRIORITY`), and unrecorded performance handling.
* `tests/test_source_mapping.py`: Validates anti-hallucination source ID filtering and deterministic cache fingerprinting.

---

## 10. Responsible Educational AI Disclaimer

> MediCompass AI is an educational study intelligence and concept-retention system designed for entry test preparation. It does not provide medical advice or medical diagnosis. Students should always verify critical curriculum outcomes with the official PMDC MDCAT / NUMS syllabus and authorized provincial textbook boards.

---

## 11. License

Released under the [MIT License](LICENSE).
