"""Async Anthropic wrapper with disk cache, retries, and tool-use support."""

import asyncio
from contextlib import suppress
from typing import Any, cast

import anthropic
from anthropic.types import Message, MessageParam, TextBlock, ToolParam, ToolUseBlock

from rageval.cache import get_cache_key, load_from_cache, save_to_cache

# Cost per 1M tokens in USD: (input, output)
_COST_PER_M: dict[str, tuple[float, float]] = {
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-5-haiku-20241022": (0.80, 4.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-5-sonnet-20240620": (3.0, 15.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-opus-4-7": (15.0, 75.0),
}
_DEFAULT_COST_PER_M: tuple[float, float] = (3.0, 15.0)

_MAX_RETRIES = 5
_BASE_BACKOFF = 1.0
_MAX_BACKOFF = 60.0
_DEFAULT_MAX_TOKENS = 4096


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    cost_in, cost_out = _COST_PER_M.get(model, _DEFAULT_COST_PER_M)
    return (input_tokens * cost_in + output_tokens * cost_out) / 1_000_000


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        max_concurrency: int = 5,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._sem = asyncio.Semaphore(max_concurrency)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0

    async def complete(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.0,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        no_cache: bool = False,
    ) -> dict[str, Any]:
        """Call the Anthropic API.

        The returned dict always contains:
          - content: concatenated text from any text blocks (may be "")
          - tool_calls: list of {"id", "name", "input"} dicts from tool_use blocks
          - model, input_tokens, output_tokens, cost_usd, cached
        """
        tool_schema: dict[str, Any] | None = None
        if tools is not None:
            tool_schema = {"tools": tools, "tool_choice": tool_choice}
        key = get_cache_key(model, system, user, temperature, tool_schema=tool_schema)

        if not no_cache:
            cached = load_from_cache(key)
            if cached is not None:
                cached = dict(cached)
                cached["cached"] = True
                # Ensure tool_calls key exists for backward compat with older cache entries.
                cached.setdefault("tool_calls", [])
                return cached

        async with self._sem:
            response = await self._call_with_retry(
                model, system, user, temperature, tools=tools, tool_choice=tool_choice
            )

        input_tokens: int = response.usage.input_tokens
        output_tokens: int = response.usage.output_tokens
        cost = _estimate_cost(model, input_tokens, output_tokens)

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        result: dict[str, Any] = {
            "content": "".join(text_parts),
            "tool_calls": tool_calls,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        }

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        if not no_cache:
            save_to_cache(key, result)

        result["cached"] = False
        return result

    async def _call_with_retry(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> Message:
        messages: list[MessageParam] = [{"role": "user", "content": user}]

        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "max_tokens": _DEFAULT_MAX_TOKENS,
                    "temperature": temperature,
                    "system": system,
                    "messages": messages,
                }
                if tools is not None:
                    kwargs["tools"] = cast(list[ToolParam], tools)
                    if tool_choice is not None:
                        kwargs["tool_choice"] = tool_choice
                message: Message = await self._client.messages.create(**kwargs)
                return message
            except anthropic.RateLimitError as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                retry_after: float | None = None
                ra_header = exc.response.headers.get("retry-after")
                if ra_header is not None:
                    with suppress(ValueError):
                        retry_after = float(ra_header)
                delay = (
                    retry_after
                    if retry_after is not None
                    else min(_BASE_BACKOFF * (2**attempt), _MAX_BACKOFF)
                )
                await asyncio.sleep(delay)

        raise RuntimeError("unreachable")
