"""OpenAI-compatible LLM access for the Teacher Assistant.

LiteLLM owns routing from the configured model alias to the model served on
the DGX Spark. Keeping client construction here gives every caller the same
endpoint, credentials, timeout behavior, and useful error messages.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
import re

from openai import OpenAI

from teacher_assistant import settings


@dataclass(frozen=True)
class ChatResult:
    content: str
    prompt_tokens: int = 0


@dataclass(frozen=True)
class StreamChunk:
    content: str = ""
    prompt_tokens: int = 0


def _safe_schema_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64] or "response"


@lru_cache(maxsize=1)
def client() -> OpenAI:
    if not settings.LLM_API_KEY:
        raise RuntimeError(
            "LLM_API_KEY is not configured for LLM endpoint "
            f"{settings.LLM_BASE_URL}"
        )
    return OpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )


def _raise_request_error(exc: Exception) -> None:
    detail = str(exc)
    if settings.LLM_API_KEY:
        detail = detail.replace(settings.LLM_API_KEY, "[REDACTED]")
    raise RuntimeError(
        f"LLM request to {settings.LLM_BASE_URL} failed: {detail}"
    ) from exc


def check_connection() -> None:
    """Verify LiteLLM authentication and routing without requiring output."""
    try:
        client().models.list()
    except Exception as exc:
        _raise_request_error(exc)


def chat(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float | None = None,
    response_schema: dict | None = None,
    schema_name: str = "response",
) -> ChatResult:
    request = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": settings.LLM_REASONING_EFFORT,
    }
    if temperature is not None:
        request["temperature"] = temperature
    if response_schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": _safe_schema_name(schema_name),
                "schema": response_schema,
            },
        }
    try:
        response = client().chat.completions.create(**request)
    except Exception as exc:
        _raise_request_error(exc)
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError(
            f"LLM request to {settings.LLM_BASE_URL} returned no visible content. "
            "The model may have used its completion budget for hidden reasoning."
        )
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    return ChatResult(content=content, prompt_tokens=prompt_tokens)


def stream_chat(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float | None = None,
) -> Iterator[StreamChunk]:
    request = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": settings.LLM_REASONING_EFFORT,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        request["temperature"] = temperature
    emitted_content = False
    try:
        stream = client().chat.completions.create(**request)
        for chunk in stream:
            content = ""
            if chunk.choices:
                content = chunk.choices[0].delta.content or ""
            prompt_tokens = chunk.usage.prompt_tokens if chunk.usage else 0
            if content:
                emitted_content = True
            # Reasoning models can emit hundreds of deltas containing only
            # `reasoning_content`. Do not turn those into empty Gradio updates.
            if content or prompt_tokens:
                yield StreamChunk(content=content, prompt_tokens=prompt_tokens)
    except Exception as exc:
        _raise_request_error(exc)
    if not emitted_content:
        raise RuntimeError(
            f"LLM request to {settings.LLM_BASE_URL} returned no visible content. "
            "The model may have used its completion budget for hidden reasoning."
        )
