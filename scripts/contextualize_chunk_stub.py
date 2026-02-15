#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    source_path = payload.get("source_path", "unknown")
    chunk_index = payload.get("chunk_index", 1)
    total_chunks = payload.get("total_chunks", 1)
    chunk_text = str(payload.get("chunk_text", "")).strip().replace("\n", " ")
    max_chars = int(payload.get("max_context_chars", 320))
    context = (
        f"File {source_path}, chunk {chunk_index}/{total_chunks}. "
        f"Topic summary: {chunk_text[:220]}"
    )[:max_chars]
    print(json.dumps({"context": context}))


if __name__ == "__main__":
    main()
