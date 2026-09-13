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
