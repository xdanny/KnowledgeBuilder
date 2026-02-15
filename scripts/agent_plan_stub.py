#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    backend = payload.get("backend", "aws_kb")

    if backend == "faiss":
        plan = {
            "backend": "faiss",
            "embedding_provider": "sentence_transformers",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size_chars": 1600,
            "chunk_overlap_chars": 220,
            "notes": "Stub planner selected local FAISS settings.",
        }
    else:
        plan = {
            "backend": "aws_kb",
            "embedding_provider": "bedrock_kb_managed",
            "embedding_model": "managed_by_knowledge_base",
            "chunk_size_chars": 1800,
            "chunk_overlap_chars": 220,
            "notes": "Stub planner selected Bedrock KB settings.",
        }
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
