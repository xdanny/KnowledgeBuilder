from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from kb_indexer.settings import AppSettings


@dataclass
class IngestionPlan:
    backend: str
    embedding_provider: str
    embedding_model: str
    chunk_size_chars: int
    chunk_overlap_chars: int
    notes: str = ""


def _heuristic_plan(settings: AppSettings) -> IngestionPlan:
    if settings.backend.type == "faiss":
        provider = settings.backend.faiss.embedding_provider
        model = settings.backend.faiss.embedding_model
        return IngestionPlan(
            backend="faiss",
            embedding_provider=provider,
            embedding_model=model,
            chunk_size_chars=settings.indexing.contextual_chunk_size_chars,
            chunk_overlap_chars=settings.indexing.contextual_chunk_overlap_chars,
            notes="Heuristic plan selected based on configured FAISS backend.",
        )
    return IngestionPlan(
        backend="aws_kb",
        embedding_provider="bedrock_kb_managed",
        embedding_model="managed_by_knowledge_base",
        chunk_size_chars=1800,
        chunk_overlap_chars=220,
        notes="Heuristic plan selected for Bedrock Knowledge Base ingestion.",
    )


def _external_command_plan(settings: AppSettings, context: dict[str, Any]) -> IngestionPlan:
    command = settings.planner.external_command.command
    if not command:
        raise ValueError("planner.external_command.command is required for external planner mode.")

    payload = {
        "backend": settings.backend.type,
        "repositories": [r.__dict__ for r in settings.repositories],
        "indexing": settings.indexing.__dict__,
        "faiss": settings.backend.faiss.__dict__,
        "aws": settings.aws.__dict__,
        "context": context,
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
    return IngestionPlan(
        backend=str(data.get("backend", settings.backend.type)),
        embedding_provider=str(
            data.get("embedding_provider", settings.backend.faiss.embedding_provider)
        ),
        embedding_model=str(data.get("embedding_model", settings.backend.faiss.embedding_model)),
        chunk_size_chars=int(
            data.get("chunk_size_chars", settings.indexing.contextual_chunk_size_chars)
        ),
        chunk_overlap_chars=int(
            data.get("chunk_overlap_chars", settings.indexing.contextual_chunk_overlap_chars)
        ),
        notes=str(data.get("notes", "External planner applied.")),
    )


def build_plan(settings: AppSettings, context: dict[str, Any]) -> IngestionPlan:
    if settings.planner.mode == "external_command":
        return _external_command_plan(settings, context)
    return _heuristic_plan(settings)
