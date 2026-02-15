from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kb_indexer.backends.base import IndexBackend
from kb_indexer.contextual_retrieval import contextualize_document, tokenize
from kb_indexer.embeddings import Embedder, HashEmbedder, SentenceTransformerEmbedder
from kb_indexer.extractor import KnowledgeDocument
from kb_indexer.settings import AppSettings


def _load_faiss_module() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is not installed. Run: uv sync --extra faiss") from exc
    return faiss


def _load_bm25_module() -> Any:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("rank-bm25 is not installed. Run: uv sync --extra faiss") from exc
    return BM25Okapi


class FaissBackend(IndexBackend):
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.faiss_settings = settings.backend.faiss
        self.faiss = _load_faiss_module()
        self.index_path = Path(self.faiss_settings.index_path)
        self.metadata_path = Path(self.faiss_settings.metadata_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = self._build_embedder()
        self.index = self._load_or_create_index()
        self.meta = self._load_metadata()
        self.docs_uploaded = 0
        self.docs_deleted = 0
        self.chunks_uploaded = 0

        self._bm25_class = None
        self._bm25 = None
        self._bm25_chunk_ids: list[int] = []
        self._bm25_ready = False

    def _build_embedder(self) -> Embedder:
        if self.faiss_settings.embedding_provider == "sentence_transformers":
            return SentenceTransformerEmbedder(
                model_name=self.faiss_settings.embedding_model,
                normalize=self.faiss_settings.normalize_embeddings,
            )
        return HashEmbedder(dimension=self.faiss_settings.embedding_dimension)

    def _load_or_create_index(self) -> Any:
        if self.index_path.exists():
            return self.faiss.read_index(str(self.index_path))
        base = self.faiss.IndexFlatIP(self.faiss_settings.embedding_dimension)
        return self.faiss.IndexIDMap2(base)

    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {"next_id": 1, "doc_to_ids": {}, "id_to_meta": {}}
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        data.setdefault("next_id", 1)
        data.setdefault("doc_to_ids", {})
        data.setdefault("id_to_meta", {})
        return data

    def _save(self) -> None:
        self.faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _delete_doc_internal(self, source_path: str) -> None:
        ids = self.meta["doc_to_ids"].pop(source_path, [])
        if not ids:
            return
        np_ids = np.asarray(ids, dtype=np.int64)
        self.index.remove_ids(np_ids)
        for idx in ids:
            self.meta["id_to_meta"].pop(str(idx), None)
        self.docs_deleted += 1
        self._bm25_ready = False

    def delete_documents(self, source_paths: list[str]) -> None:
        for path in source_paths:
            self._delete_doc_internal(path)

    def delete_by_prefix(self, prefixes: list[str]) -> None:
        if not prefixes:
            return
        all_paths = list(self.meta["doc_to_ids"].keys())
        for path in all_paths:
            if any(path.startswith(prefix) for prefix in prefixes):
                self._delete_doc_internal(path)

    def _ensure_dimension(self, vectors: np.ndarray) -> None:
        if self.index.ntotal == 0 and vectors.shape[1] != self.index.d:
            base = self.faiss.IndexFlatIP(vectors.shape[1])
            self.index = self.faiss.IndexIDMap2(base)
            return
        if self.index.ntotal > 0 and vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension {vectors.shape[1]} does not match existing index "
                f"dimension {self.index.d}. Rebuild with --full and a fresh index path."
            )

    def _chunk_document(self, doc: KnowledgeDocument) -> list[dict[str, Any]]:
        chunks = contextualize_document(
            source_path=doc.source_path,
            document_text=doc.content,
            settings=self.settings,
        )
        out: list[dict[str, Any]] = []
        for chunk in chunks:
            out.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                    "raw_text": chunk.raw_text,
                    "context": chunk.context,
                    "contextual_text": chunk.contextual_text,
                    "kind": doc.kind,
                }
            )
        return out

    def upsert_documents(self, docs: list[KnowledgeDocument]) -> None:
        for doc in docs:
            self._delete_doc_internal(doc.source_path)
            chunk_rows = self._chunk_document(doc)
            vectors = self.embedder.encode([row["contextual_text"] for row in chunk_rows]).astype(
                np.float32
            )
            self._ensure_dimension(vectors)
            ids: list[int] = []
            for row in chunk_rows:
                idx = int(self.meta["next_id"])
                self.meta["next_id"] = idx + 1
                ids.append(idx)
                self.meta["id_to_meta"][str(idx)] = row
            np_ids = np.asarray(ids, dtype=np.int64)
            self.index.add_with_ids(vectors, np_ids)
            self.meta["doc_to_ids"][doc.source_path] = ids
            self.docs_uploaded += 1
            self.chunks_uploaded += len(chunk_rows)
            self._bm25_ready = False

    def _ensure_bm25(self) -> None:
        if not self.faiss_settings.bm25_enabled:
            return
        if self._bm25_ready:
            return
        if self._bm25_class is None:
            self._bm25_class = _load_bm25_module()

        chunk_ids = sorted(int(key) for key in self.meta["id_to_meta"].keys())
        corpus = [
            tokenize(self.meta["id_to_meta"][str(chunk_id)].get("contextual_text", ""))
            for chunk_id in chunk_ids
        ]
        self._bm25_chunk_ids = chunk_ids
        self._bm25 = self._bm25_class(corpus) if corpus else None
        self._bm25_ready = True

    def _dense_candidates(self, query: str, limit: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        query_vec = self.embedder.encode([query]).astype(np.float32)
        k = min(max(limit, 1), self.index.ntotal)
        scores, ids = self.index.search(query_vec, k)
        out: list[tuple[int, float]] = []
        for score, chunk_id in zip(scores[0], ids[0]):
            if chunk_id < 0:
                continue
            out.append((int(chunk_id), float(score)))
        return out

    def _bm25_candidates(self, query: str, limit: int) -> list[tuple[int, float]]:
        self._ensure_bm25()
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][:limit]
        return [
            (self._bm25_chunk_ids[int(i)], float(scores[int(i)]))
            for i in top_idx
            if scores[int(i)] > 0
        ]

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        candidate_k = max(self.faiss_settings.initial_retrieval_k, top_k)
        dense = self._dense_candidates(query, candidate_k)
        bm25 = self._bm25_candidates(query, candidate_k) if self.faiss_settings.bm25_enabled else []

        rrf_k = max(self.faiss_settings.rrf_k, 1)
        fused: dict[int, dict[str, float]] = {}
        for rank, (chunk_id, score) in enumerate(dense, start=1):
            row = fused.setdefault(chunk_id, {"rrf": 0.0, "dense_score": score, "bm25_score": 0.0})
            row["rrf"] += 1.0 / (rrf_k + rank)
            row["dense_score"] = score
        for rank, (chunk_id, score) in enumerate(bm25, start=1):
            row = fused.setdefault(chunk_id, {"rrf": 0.0, "dense_score": 0.0, "bm25_score": score})
            row["rrf"] += 1.0 / (rrf_k + rank)
            row["bm25_score"] = score

        query_tokens = set(tokenize(query))
        scored: list[dict[str, Any]] = []
        for chunk_id, scores in fused.items():
            meta = self.meta["id_to_meta"].get(str(chunk_id), {})
            contextual = meta.get("contextual_text", "")
            chunk_tokens = set(tokenize(contextual))
            overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            final = scores["rrf"]
            if self.faiss_settings.rerank_enabled:
                final = 0.8 * scores["rrf"] + 0.2 * overlap
            scored.append(
                {
                    "chunk_id": chunk_id,
                    "source_path": meta.get("source_path"),
                    "chunk_index": meta.get("chunk_index"),
                    "kind": meta.get("kind"),
                    "context": meta.get("context"),
                    "raw_text": meta.get("raw_text"),
                    "contextual_text": contextual,
                    "dense_score": scores["dense_score"],
                    "bm25_score": scores["bm25_score"],
                    "overlap_score": overlap,
                    "final_score": final,
                }
            )

        scored.sort(key=lambda item: item["final_score"], reverse=True)
        return scored[:top_k]

    def finalize(self) -> dict[str, Any]:
        self._save()
        return {
            "backend": "faiss",
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "uploaded_docs": self.docs_uploaded,
            "deleted_docs": self.docs_deleted,
            "uploaded_chunks": self.chunks_uploaded,
            "bm25_enabled": self.faiss_settings.bm25_enabled,
        }
