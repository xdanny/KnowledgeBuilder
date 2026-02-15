from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from kb_indexer.extractor import KnowledgeDocument


class IndexBackend(ABC):
    @abstractmethod
    def upsert_documents(self, docs: list[KnowledgeDocument]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_documents(self, source_paths: list[str]) -> None:
        raise NotImplementedError

    def delete_by_prefix(self, prefixes: list[str]) -> None:
        return None

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        raise NotImplementedError
