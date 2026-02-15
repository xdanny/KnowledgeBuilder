from __future__ import annotations

from typing import Any

from kb_indexer.backends.aws_kb import AwsKbBackend
from kb_indexer.backends.base import IndexBackend
from kb_indexer.backends.faiss_local import FaissBackend
from kb_indexer.settings import AppSettings


def build_backend(settings: AppSettings, *, s3_client: Any, bedrock_agent_client: Any) -> IndexBackend:
    if settings.backend.type == "faiss":
        return FaissBackend(settings=settings)
    return AwsKbBackend(s3_client=s3_client, bedrock_agent_client=bedrock_agent_client, settings=settings)
