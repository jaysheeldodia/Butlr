from __future__ import annotations

from openai import OpenAI

from config.settings import OPENAI_API_KEY


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding for the given text.
    Use OpenAI text-embedding-3-small via the openai SDK.
    Returns a list of floats.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return list(response.data[0].embedding)
