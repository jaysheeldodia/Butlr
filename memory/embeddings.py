from __future__ import annotations

from config.settings import EMBEDDING_MODEL

_model = None


def _get_model():
    """Load the local embedding model once and reuse it.

    Uses sentence-transformers with BAAI/bge-small-en-v1.5 (MIT licence,
    384 dims). Lazy import so tests and non-embedding paths never need the
    model installed. Email bodies embedded here stay on this machine.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    """
    Generate a local embedding for the given text.
    Returns a list of floats of length EMBEDDING_DIM (384 by default).
    No network call is made for the text itself.
    """
    vector = _get_model().encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [float(x) for x in vector]
