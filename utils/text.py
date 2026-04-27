from __future__ import annotations

from pathlib import Path
from typing import List

from docx import Document
from pypdf import PdfReader


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    text = clean_text(text)
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def read_text_file(path: str) -> str:
    return clean_text(Path(path).read_text(encoding="utf-8", errors="ignore"))


def read_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))


def read_docx_file(path: str) -> str:
    doc = Document(path)
    lines = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return clean_text("\n".join(lines))


def clean_text(text: str) -> str:
    normalized = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.strip() for line in normalized.split("\n"))
    normalized = "\n".join(line for line in normalized.split("\n") if line)
    return normalized.strip()
