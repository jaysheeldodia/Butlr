from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from memory.embeddings import embed_text
from config.settings import EMBEDDING_DIM
from utils.logger import logger

_STORE_DIR = Path(__file__).resolve().parent.parent / ".butlr_store"
_INDEX_PATH = _STORE_DIR / "faiss.index"
_META_PATH = _STORE_DIR / "metadata.json"


class VectorStore:
    def __init__(self, dim: int | None = None):
        self.dim = dim if dim is not None else EMBEDDING_DIM
        _STORE_DIR.mkdir(exist_ok=True)

        if _INDEX_PATH.exists() and _META_PATH.exists():
            try:
                loaded = faiss.read_index(str(_INDEX_PATH))
                # Reject if dimension doesn't match (e.g. after embedding model change).
                if loaded.d == self.dim:
                    self.index = loaded
                    self.metadata: list[dict[str, Any]] = json.loads(_META_PATH.read_text())
                    return
                logger.warning(
                    "Stored FAISS index dim (%d) differs from configured EMBEDDING_DIM (%d). "
                    "Rebuilding an empty index — old vectors are incompatible.",
                    loaded.d,
                    self.dim,
                )
            except Exception as exc:
                logger.warning("Could not load FAISS index, rebuilding: %s", exc)

        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []

    def add(self, text: str, metadata: dict) -> None:
        vector = np.array([embed_text(text)], dtype="float32")
        self.index.add(vector)
        self.metadata.append(metadata)

    def save(self) -> None:
        try:
            faiss.write_index(self.index, str(_INDEX_PATH))
            _META_PATH.write_text(json.dumps(self.metadata))
        except Exception as exc:
            logger.exception("Could not save vector store: %s", exc)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([embed_text(query)], dtype="float32")
        _, indices = self.index.search(query_vector, top_k)

        results: list[dict] = []
        for idx in indices[0]:
            if idx == -1 or idx >= len(self.metadata):
                continue
            results.append(self.metadata[idx])
        return results
