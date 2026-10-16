from __future__ import annotations

import json
import re

from anthropic import Anthropic
from groq import Groq
from openai import OpenAI

from config.settings import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    LLM_PROVIDER,
    OPENAI_API_KEY,
)
from utils.logger import logger


def parse_json_response(raw_response: str) -> object:
    """
    Parse a model response that is expected to contain JSON.
    Handles empty responses and common markdown code-fence wrapping.
    """
    if not raw_response or not raw_response.strip():
        raise ValueError("LLM response was empty")

    normalized = raw_response.strip()
    fence_match = re.match(
        r"^```(?:json)?\s*(.*?)\s*```$",
        normalized,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        normalized = fence_match.group(1).strip()

    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        start_positions = [pos for pos in (normalized.find("{"), normalized.find("[")) if pos >= 0]
        end_positions = [pos for pos in (normalized.rfind("}"), normalized.rfind("]")) if pos >= 0]
        if not start_positions or not end_positions:
            raise

        candidate = normalized[min(start_positions) : max(end_positions) + 1].strip()
        return json.loads(candidate)


def chat(system_prompt: str, user_prompt: str) -> str:
    """
    Route to the correct LLM provider based on LLM_PROVIDER in settings.
    Returns the assistant's text response as a plain string.
    """
    try:
        if LLM_PROVIDER == "local":
            # OpenAI-compatible endpoint: llama-server / Ollama on this
            # machine, or a remote OpenAI-compatible API (e.g. OpenRouter)
            # when LOCAL_LLM_URL points there. Lazy import so tests and cloud
            # providers never need httpx at module load time.
            import json as _json

            import httpx

            from config.settings import (
                LOCAL_LLM_API_KEY,
                LOCAL_LLM_EXTRA_BODY,
                LOCAL_LLM_MODEL,
                LOCAL_LLM_TIMEOUT_SECONDS,
                LOCAL_LLM_URL,
            )

            headers: dict[str, str] = {}
            if LOCAL_LLM_API_KEY:
                headers["Authorization"] = f"Bearer {LOCAL_LLM_API_KEY}"

            body: dict = {
                "model": LOCAL_LLM_MODEL,
                "max_tokens": 1024,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            # Optional provider-specific extras (e.g. OpenRouter ZDR routing:
            # {"provider": {"zdr": true, "data_collection": "deny"}}).
            if LOCAL_LLM_EXTRA_BODY:
                body.update(_json.loads(LOCAL_LLM_EXTRA_BODY))

            response = httpx.post(
                LOCAL_LLM_URL,
                json=body,
                headers=headers,
                timeout=LOCAL_LLM_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                # Surface the server's own error message — llama-server and
                # OpenRouter both explain 400s in the body (e.g. context
                # overflow, unknown model). Without this you get only a bare
                # '400 Bad Request' and have to guess.
                detail = response.text[:500]
                logger.error(
                    "Local LLM endpoint returned %s for %s: %s",
                    response.status_code,
                    LOCAL_LLM_URL,
                    detail,
                )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"] or ""

        if LLM_PROVIDER == "openai":
            client = OpenAI(api_key=OPENAI_API_KEY, max_retries=3)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""

        if LLM_PROVIDER == "anthropic":
            client = Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(
                block.text for block in response.content if getattr(block, "text", None)
            )

        if LLM_PROVIDER == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"{system_prompt}\n\nUser:\n{user_prompt}"
            )
            return getattr(response, "text", "") or ""

        if LLM_PROVIDER == "groq":
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""

        raise ValueError(
            "Unsupported LLM_PROVIDER. Expected one of: openai, anthropic, gemini, groq."
        )
    except Exception as exc:
        logger.exception("LLM request failed for provider '%s': %s", LLM_PROVIDER, exc)
        raise
