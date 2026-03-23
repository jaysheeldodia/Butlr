from __future__ import annotations

from memory.vector_store import VectorStore
from utils.logger import logger


def retrieve_context(query: str, vector_store: VectorStore) -> str:
    results = vector_store.search(query, top_k=3)
    if not results:
        return ""
    logger.info("RAG retrieved %d context chunk(s)", len(results))

    formatted_chunks: list[str] = []
    for item in results:
        source = item.get("source", "unknown")
        title = item.get("title", "Untitled")
        content = item.get("content", "")
        formatted_chunks.append(f"Source: {source}\nTitle: {title}\nContent: {content}")
    return "\n\n".join(formatted_chunks)
