from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(Embedder):
    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big", signed=False)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.dimension).astype(np.float32)
            vectors[i] = vector
        return _l2_normalize(vectors)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, normalize: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: uv sync --extra local-embeddings"
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        return vectors


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
