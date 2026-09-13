"""
MediCompass AI - Data Ingestion & Chunking Module
Extracts, cleans, chunks, and attaches metadata to authoritative Pakistani medical entry test
sources (Punjab textbooks, Federal textbooks, PMDC syllabus, and verified past papers).
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
import re
from typing import Any

import pypdf

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a discrete semantic chunk of an authoritative study source."""

    id: str
    text: str
    title: str
    source_type: str  # 'Punjab Book', 'Federal Book', 'Syllabus', 'Past Paper', 'Study Source'
    subject: str  # 'Biology', 'Chemistry', 'Physics'
    exam: list[str]  # e.g. ['MDCAT', 'NUMS']
    chapter: str
    section: str
    page: str  # 1-indexed string or 'unknown'
    year: str  # e.g. '2023' or 'N/A'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_text(raw_text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not raw_text:
        return ""
    # Replace multiple spaces/newlines with clean spacing
    cleaned = re.sub(r"\r\n|\r", "\n", raw_text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_pdf_pages(file_path: str) -> list[dict[str, Any]]:
    """
    Extract text per page using pypdf.
    If page number cannot be determined, marks page as 'unknown'.
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return []

    pages = []
    try:
        reader = pypdf.PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            page_num_str = str(idx + 1) if idx is not None else "unknown"
            cleaned = clean_text(extracted)
            if cleaned:
                pages.append(
                    {
                        "page": page_num_str,
                        "text": cleaned,
                    }
                )
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        # Fallback with unknown page if catastrophic failure
        pages.append(
            {
                "page": "unknown",
                "text": "",
            }
        )

    return pages


def chunk_text(
    text: str, chunk_size: int = 1200, overlap: int = 200
) -> list[str]:
    """
    Produce semantic-friendly text chunks with overlap.
    Splits along paragraphs or sentence boundaries where possible.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If not at the end of the text, try to find a natural boundary
        if end < text_len:
            # Look for double newline first, then single newline, then period
            boundary = text.rfind("\n\n", start + overlap, end)
            if boundary == -1:
                boundary = text.rfind("\n", start + overlap, end)
            if boundary == -1:
                boundary = text.rfind(". ", start + overlap, end)
            if boundary != -1:
                end = boundary + (2 if text[boundary : boundary + 2] in ("\n\n", ". ") else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start position by chunk_size - overlap
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


def compute_kb_fingerprint(sources_dir: str, manifest_path: str = "") -> str:
    """
    Computes a deterministic hash fingerprint based on files, sizes, and mtimes
    in sources_dir and the manifest to avoid rebuilding index unnecessarily.
    """
    hasher = hashlib.sha256()

    file_entries = []
    if os.path.exists(sources_dir):
        for root, _, files in os.walk(sources_dir):
            for f in sorted(files):
                if f.lower().endswith(".pdf"):
                    full_path = os.path.join(root, f)
                    try:
                        stat = os.stat(full_path)
                        file_entries.append((f, stat.st_size, stat.st_mtime))
                    except OSError:
                        pass

    if manifest_path and os.path.exists(manifest_path):
        try:
            stat = os.stat(manifest_path)
            file_entries.append((os.path.basename(manifest_path), stat.st_size, stat.st_mtime))
        except OSError:
            pass

    for entry in sorted(file_entries):
        hasher.update(f"{entry[0]}:{entry[1]}:{entry[2]}".encode("utf-8"))

    return hasher.hexdigest()[:16]


def load_manifest(manifest_path: str) -> dict[str, dict[str, Any]]:
    """Load source manifest JSON and index by filename."""
    manifest_map = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    manifest_map[item.get("file")] = item
        except Exception as e:
            logger.error(f"Failed to load manifest at {manifest_path}: {e}")
    return manifest_map


def load_past_papers(past_papers_path: str) -> list[dict[str, Any]]:
    """Load verified past paper logs from JSON."""
    if os.path.exists(past_papers_path):
        try:
            with open(past_papers_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load past papers at {past_papers_path}: {e}")
    return []


def build_knowledge_base_chunks(
    sources_dir: str,
    manifest_path: str,
    past_papers_path: str = "",
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Ingests all PDFs from sources_dir, applies metadata from manifest,
    ingests verified past paper concepts, and creates DocumentChunk items.
    """
    chunks: list[DocumentChunk] = []
    manifest_map = load_manifest(manifest_path)

    # 1. Process PDF Files
    if os.path.exists(sources_dir):
        pdf_files = [f for f in os.listdir(sources_dir) if f.lower().endswith(".pdf")]
        for pdf_file in sorted(pdf_files):
            file_path = os.path.join(sources_dir, pdf_file)
            meta = manifest_map.get(pdf_file, {})

            source_type = meta.get("source_type", "Study Source")
            subject = meta.get("subject", "Biology")
            exam = meta.get("exam", ["MDCAT", "NUMS"])
            chapter = meta.get("chapter", "")
            title = meta.get("edition", pdf_file.replace(".pdf", "").replace("_", " "))

            pages_data = extract_pdf_pages(file_path)
            for p_info in pages_data:
                page_str = p_info["page"]
                page_text = p_info["text"]

                raw_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
                for c_idx, c_text in enumerate(raw_chunks):
                    # Deterministic Chunk ID
                    chunk_id = f"{pdf_file.replace('.pdf', '')}_p{page_str}_c{c_idx}"
                    chunk_obj = DocumentChunk(
                        id=chunk_id,
                        text=c_text,
                        title=title,
                        source_type=source_type,
                        subject=subject,
                        exam=exam,
                        chapter=chapter,
                        section="",
                        page=page_str if page_str != "unknown" else "unknown",
                        year=meta.get("year", "N/A"),
                    )
                    chunks.append(chunk_obj)

    # 2. Ingest Verified Past Papers as authoritative concept evidence chunks
    if past_papers_path and os.path.exists(past_papers_path):
        past_papers = load_past_papers(past_papers_path)
        for idx, paper in enumerate(past_papers):
            if not paper.get("verified", False):
                continue

            year_str = str(paper.get("year", "Historical"))
            exam_name = paper.get("exam", "MDCAT")
            concept = paper.get("concept", "")
            q_id = paper.get("question_id", f"PP-{year_str}-{idx}")
            excerpt = paper.get("source_excerpt", "")
            summary = paper.get("question_summary", "")

            combined_text = (
                f"Historical Past Paper Concept: {concept}\n"
                f"Exam: {exam_name} {year_str}\n"
                f"Question Reference: {q_id}\n"
                f"Summary: {summary}\n"
                f"Excerpt: {excerpt}"
            )

            chunk_id = f"pastpaper_{exam_name}_{year_str}_{q_id}"
            chunk_obj = DocumentChunk(
                id=chunk_id,
                text=combined_text,
                title=f"{exam_name} {year_str} Past Paper ({q_id})",
                source_type="Past Paper",
                subject=paper.get("subject", "Biology"),
                exam=[exam_name],
                chapter="",
                section=f"Question {q_id}",
                page="unknown",
                year=year_str,
            )
            chunks.append(chunk_obj)

    return chunks


def sync_google_drive_public_folders(
    folder_urls: list[str], target_dir: str
) -> list[str]:
    """
    Downloads documents from public Google Drive folder links using gdown.
    Caches downloaded files so they are not re-downloaded on every run.
    Note: Private Google Drive requires OAuth / Service Account credentials.
    """
    downloaded_files = []
    if not folder_urls:
        return downloaded_files

    try:
        import gdown  # Lazy import
    except ImportError:
        logger.warning("gdown library not installed. Cannot sync Google Drive folders.")
        return downloaded_files

    os.makedirs(target_dir, exist_ok=True)

    for url in folder_urls:
        url = url.strip()
        if not url or url.startswith("#"):
            continue

        try:
            logger.info(f"Syncing Google Drive public folder: {url}")
            # Use gdown folder download to target_dir with remaining_ok=True
            res = gdown.download_folder(url, output=target_dir, quiet=True, use_cookies=False)
            if res:
                downloaded_files.extend(res)
        except Exception as e:
            logger.error(f"Failed to sync Google Drive folder {url}: {e}")

    return downloaded_files
