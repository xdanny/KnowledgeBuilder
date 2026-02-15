from __future__ import annotations

import json
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
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


def _cache_key(
    *,
    mode: str,
    model: str | None,
    source_path: str,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str,
    max_chars: int,
) -> str:
    payload = (
        f"{mode}|{model or ''}|{source_path}|{chunk_index}|{total_chunks}|{max_chars}|{chunk_text}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_context_cache(cache_path: str) -> dict[str, str]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _save_context_cache(cache_path: str, cache: dict[str, str]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=True, indent=2), encoding="utf-8")


def _llm_prompt(
    *,
    source_path: str,
    chunk_index: int,
    total_chunks: int,
    document_text: str,
    chunk_text: str,
    max_chars: int,
) -> str:
    doc_prefix = document_text[:2400]
    return (
        "Generate short retrieval context for a code/document chunk.\n"
        f"Hard limit: {max_chars} characters.\n"
        "Return plain text only, no markdown or JSON.\n"
        "Include only high-signal facts:\n"
        "- main purpose of this chunk\n"
        "- key entities (tables, jobs, metrics, functions/classes)\n"
        "- dependencies and dataflow clues\n\n"
        f"Source path: {source_path}\n"
        f"Chunk: {chunk_index}/{total_chunks}\n\n"
        "Document prefix:\n"
        f"{doc_prefix}\n\n"
        "Chunk text:\n"
        f"{chunk_text}\n"
    )


def _bedrock_context(
    *,
    settings: AppSettings,
    source_path: str,
    chunk_index: int,
    total_chunks: int,
    document_text: str,
    chunk_text: str,
    max_chars: int,
) -> str:
    model_id = settings.indexing.contextualizer_model
    if not model_id:
        raise ValueError("indexing.contextualizer_model is required when contextualizer_mode=bedrock.")
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for contextualizer_mode=bedrock.") from exc

    client = boto3.client("bedrock-runtime", region_name=settings.aws.region)
    prompt = _llm_prompt(
        source_path=source_path,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        document_text=document_text,
        chunk_text=chunk_text,
        max_chars=max_chars,
    )
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={
            "temperature": settings.indexing.contextualizer_temperature,
            "maxTokens": settings.indexing.contextualizer_max_tokens,
        },
    )
    content = response.get("output", {}).get("message", {}).get("content", [])
    text_parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
    context = " ".join(part.strip() for part in text_parts if part.strip()).strip()
    if not context:
        raise ValueError("Bedrock contextualizer returned empty context.")
    return context[:max_chars]


def _litellm_context(
    *,
    settings: AppSettings,
    source_path: str,
    chunk_index: int,
    total_chunks: int,
    document_text: str,
    chunk_text: str,
    max_chars: int,
) -> str:
    model = settings.indexing.contextualizer_model
    if not model:
        raise ValueError("indexing.contextualizer_model is required when contextualizer_mode=litellm.")
    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError(
            "litellm is not installed. Install it with: uv add litellm"
        ) from exc

    prompt = _llm_prompt(
        source_path=source_path,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        document_text=document_text,
        chunk_text=chunk_text,
        max_chars=max_chars,
    )
    response = completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create concise contextual retrieval annotations for code and data docs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=settings.indexing.contextualizer_max_tokens,
        temperature=settings.indexing.contextualizer_temperature,
    )
    if hasattr(response, "choices"):
        choices = getattr(response, "choices", [])
    elif isinstance(response, dict):
        choices = response.get("choices", [])
    else:
        choices = []
    if not choices:
        raise ValueError("LiteLLM contextualizer returned no choices.")
    message = choices[0].message if hasattr(choices[0], "message") else choices[0].get("message", {})
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        context = " ".join(part.strip() for part in text_parts if part.strip()).strip()
    else:
        context = str(content or "").strip()
    if not context:
        raise ValueError("LiteLLM contextualizer returned empty context.")
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
    cache_path = settings.indexing.contextualizer_cache_path
    context_cache = _load_context_cache(cache_path) if cache_path else {}
    cache_updated = False

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

        cache_key = _cache_key(
            mode=mode,
            model=settings.indexing.contextualizer_model,
            source_path=source_path,
            chunk_index=idx,
            total_chunks=total,
            chunk_text=raw_chunk,
            max_chars=max_chars,
        )
        cached = context_cache.get(cache_key)
        if cached:
            context = cached[:max_chars]
        else:
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
            elif mode == "bedrock":
                context = _bedrock_context(
                    settings=settings,
                    source_path=source_path,
                    chunk_index=idx,
                    total_chunks=total,
                    document_text=document_text,
                    chunk_text=raw_chunk,
                    max_chars=max_chars,
                )
            elif mode == "litellm":
                context = _litellm_context(
                    settings=settings,
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
            if cache_path:
                context_cache[cache_key] = context
                cache_updated = True
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
    if cache_path and cache_updated:
        _save_context_cache(cache_path, context_cache)
    return output
