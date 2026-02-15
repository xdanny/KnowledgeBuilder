from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from kb_indexer.settings import AppSettings


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class ContextualChunk:
    chunk_id: str
    source_path: str
    chunk_index: int
    raw_text: str
    context: str
    contextual_text: str
    metadata: dict[str, Any]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        heading_break = max(chunk.rfind("\n## "), chunk.rfind("\n### "), chunk.rfind("\n\n"))
        if heading_break > int(chunk_size * 0.4):
            end = start + heading_break
            chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def _heuristic_context(document_text: str, source_path: str, chunk: str, max_chars: int) -> str:
    header_lines = []
    for line in document_text.splitlines():
        if line.strip().startswith("#"):
            header_lines.append(line.strip())
        if len(header_lines) >= 3:
            break
    table_lines = [
        line.strip()
        for line in document_text.splitlines()
        if "Table reads:" in line or "Table writes:" in line
    ]
    snippet = " ".join(chunk.split())[:180]
    context = (
        f"Source: {source_path}. "
        f"Headers: {' | '.join(header_lines) if header_lines else 'none'}. "
        f"Table hints: {' | '.join(table_lines) if table_lines else 'none'}. "
        f"Chunk focus: {snippet}"
    )
    return context[:max_chars]


def _external_context(
    *,
    command: str,
    source_path: str,
    chunk_index: int,
    total_chunks: int,
    document_text: str,
    chunk_text: str,
    max_chars: int,
) -> str:
    payload = {
        "source_path": source_path,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "document_text": document_text,
        "chunk_text": chunk_text,
        "max_context_chars": max_chars,
    }
    result = subprocess.run(
        command,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        input=json.dumps(payload),
    )
    data = json.loads(result.stdout)
    context = str(data.get("context", "")).strip()
    if not context:
        raise ValueError("External contextualizer returned empty context.")
    return context[:max_chars]


def contextualize_document(
    *,
    source_path: str,
    document_text: str,
    settings: AppSettings,
) -> list[ContextualChunk]:
    chunk_size = settings.indexing.contextual_chunk_size_chars
    overlap = settings.indexing.contextual_chunk_overlap_chars
    max_chars = settings.indexing.contextualizer_max_context_chars
    mode = settings.indexing.contextualizer_mode
    command = settings.indexing.contextualizer_command

    chunks = split_text(document_text, chunk_size=chunk_size, overlap=overlap)
    output: list[ContextualChunk] = []
    total = len(chunks)
    for idx, raw_chunk in enumerate(chunks, start=1):
        if not settings.indexing.contextual_retrieval_enabled:
            context = ""
            contextual_text = raw_chunk
            output.append(
                ContextualChunk(
                    chunk_id=f"{source_path}::chunk_{idx}",
                    source_path=source_path,
                    chunk_index=idx,
                    raw_text=raw_chunk,
                    context=context,
                    contextual_text=contextual_text,
                    metadata={"source_path": source_path, "chunk_index": idx, "total_chunks": total},
                )
            )
            continue

        if mode == "external_command":
            if not command:
                raise ValueError(
                    "indexing.contextualizer_command is required when contextualizer_mode=external_command."
                )
            context = _external_context(
                command=command,
                source_path=source_path,
                chunk_index=idx,
                total_chunks=total,
                document_text=document_text,
                chunk_text=raw_chunk,
                max_chars=max_chars,
            )
        else:
            context = _heuristic_context(
                document_text=document_text,
                source_path=source_path,
                chunk=raw_chunk,
                max_chars=max_chars,
            )
        contextual_text = f"[Context]\n{context}\n\n[Chunk]\n{raw_chunk}"
        output.append(
            ContextualChunk(
                chunk_id=f"{source_path}::chunk_{idx}",
                source_path=source_path,
                chunk_index=idx,
                raw_text=raw_chunk,
                context=context,
                contextual_text=contextual_text,
                metadata={"source_path": source_path, "chunk_index": idx, "total_chunks": total},
            )
        )
    return output
