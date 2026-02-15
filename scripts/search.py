#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

import boto3

from kb_indexer.backends.faiss_local import FaissBackend
from kb_indexer.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search indexed knowledge with contextual hybrid retrieval."
    )
    parser.add_argument("--config", default="kb_config.yaml", help="Path to config YAML.")
    parser.add_argument("--query", required=True, help="User query text.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of results.")
    return parser.parse_args()


def search_aws_kb(settings: Any, query: str, top_k: int) -> dict[str, Any]:
    if not settings.aws.knowledge_base_id:
        raise ValueError("aws.knowledge_base_id is required for aws_kb backend search.")
    session = boto3.Session(region_name=settings.aws.region)
    runtime = session.client("bedrock-agent-runtime")
    response = runtime.retrieve(
        knowledgeBaseId=settings.aws.knowledge_base_id,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": top_k,
            }
        },
    )
    results = []
    for row in response.get("retrievalResults", []):
        content = row.get("content", {}).get("text", "")
        location = row.get("location", {})
        score = row.get("score")
        results.append(
            {
                "score": score,
                "location": location,
                "content_preview": content[:700],
            }
        )
    return {"backend": "aws_kb", "results": results}


def search_faiss(settings: Any, query: str, top_k: int) -> dict[str, Any]:
    backend = FaissBackend(settings=settings)
    results = backend.search(query=query, top_k=top_k)
    return {"backend": "faiss", "results": results}


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    if settings.backend.type == "faiss":
        output = search_faiss(settings, args.query, args.top_k)
    else:
        output = search_aws_kb(settings, args.query, args.top_k)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
