import asyncio
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest
from anthropic.types import TextBlock, ToolUseBlock

import rageval.cache as cache_module
from rageval.cache import get_cache_key, load_from_cache, save_to_cache
from rageval.llm_client import LLMClient

# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_message(
    text: str = "answer",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    msg = MagicMock()
    msg.content = [TextBlock(type="text", text=text)]
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    return msg


def _fake_tooluse_message(
    tool_name: str,
    tool_input: dict[str, object],
    text: str = "",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    msg = MagicMock()
    blocks: list[object] = []
    if text:
        blocks.append(TextBlock(type="text", text=text))
    blocks.append(
        ToolUseBlock(type="tool_use", id="tool_1", name=tool_name, input=tool_input)
    )
    msg.content = blocks
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    return msg


def _rate_limit_error(retry_after: str | None = None) -> anthropic.RateLimitError:
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=429, headers=headers, request=request)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / ".rageval_cache"
    monkeypatch.setattr(cache_module, "_CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture
def mock_create() -> Generator[AsyncMock, None, None]:
    with patch("rageval.llm_client.anthropic.AsyncAnthropic") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        create = AsyncMock(return_value=_fake_message())
        instance.messages.create = create
        yield create


# ── cache: get_cache_key ──────────────────────────────────────────────────────


def test_cache_key_is_deterministic() -> None:
    k1 = get_cache_key("model-a", "sys", "usr", 0.0)
    k2 = get_cache_key("model-a", "sys", "usr", 0.0)
    assert k1 == k2


def test_cache_key_differs_on_model() -> None:
    assert get_cache_key("model-a", "sys", "usr", 0.0) != get_cache_key(
        "model-b", "sys", "usr", 0.0
    )


def test_cache_key_differs_on_system() -> None:
    assert get_cache_key("m", "sys-a", "usr", 0.0) != get_cache_key("m", "sys-b", "usr", 0.0)


def test_cache_key_differs_on_user() -> None:
    assert get_cache_key("m", "sys", "usr-a", 0.0) != get_cache_key("m", "sys", "usr-b", 0.0)


def test_cache_key_differs_on_temperature() -> None:
    assert get_cache_key("m", "sys", "usr", 0.0) != get_cache_key("m", "sys", "usr", 1.0)


def test_cache_key_differs_on_tool_schema() -> None:
    k1 = get_cache_key("m", "sys", "usr", 0.0, tool_schema=None)
    k2 = get_cache_key("m", "sys", "usr", 0.0, tool_schema={"name": "tool"})
    assert k1 != k2


def test_cache_key_is_sha256_hex() -> None:
    key = get_cache_key("m", "sys", "usr", 0.0)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


# ── cache: save / load ────────────────────────────────────────────────────────


def test_load_missing_key_returns_none(isolated_cache: Path) -> None:
    assert load_from_cache("a" * 64) is None


def test_save_then_load_roundtrip(isolated_cache: Path) -> None:
    data = {"content": "hello", "cost_usd": 0.003, "input_tokens": 10, "output_tokens": 5}
    key = get_cache_key("model", "sys", "usr", 0.0)
    save_to_cache(key, data)
    assert load_from_cache(key) == data


def test_save_creates_directory_automatically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep_dir = tmp_path / "deeply" / "nested" / "cache"
    monkeypatch.setattr(cache_module, "_CACHE_DIR", deep_dir)
    key = get_cache_key("m", "s", "u", 0.0)
    save_to_cache(key, {"x": 1})
    assert load_from_cache(key) == {"x": 1}


def test_cache_file_uses_sharded_path(isolated_cache: Path) -> None:
    key = get_cache_key("m", "sys", "usr", 0.0)
    save_to_cache(key, {"v": 42})
    expected = isolated_cache / key[:2] / f"{key}.json"
    assert expected.exists()


def test_load_corrupt_file_returns_none(isolated_cache: Path) -> None:
    key = get_cache_key("m", "s", "u", 0.0)
    path = isolated_cache / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json", encoding="utf-8")
    assert load_from_cache(key) is None


# ── LLMClient: cache hit ──────────────────────────────────────────────────────


async def test_cache_hit_skips_api(isolated_cache: Path, mock_create: AsyncMock) -> None:
    key = get_cache_key("claude-sonnet-4-6", "sys", "usr", 0.0)
    save_to_cache(key, {
        "content": "cached answer",
        "model": "claude-sonnet-4-6",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.0001,
    })

    client = LLMClient()
    result = await client.complete("sys", "usr", "claude-sonnet-4-6")

    assert result["content"] == "cached answer"
    assert result["cached"] is True
    mock_create.assert_not_called()


# ── LLMClient: cache miss ─────────────────────────────────────────────────────


async def test_cache_miss_calls_api(isolated_cache: Path, mock_create: AsyncMock) -> None:
    mock_create.return_value = _fake_message(text="fresh")
    client = LLMClient()
    result = await client.complete("sys", "usr", "claude-sonnet-4-6")

    mock_create.assert_called_once()
    assert result["content"] == "fresh"
    assert result["cached"] is False


async def test_cache_miss_saves_result(isolated_cache: Path, mock_create: AsyncMock) -> None:
    mock_create.return_value = _fake_message(text="saved")
    client = LLMClient()
    await client.complete("sys", "usr", "claude-sonnet-4-6")

    key = get_cache_key("claude-sonnet-4-6", "sys", "usr", 0.0)
    cached = load_from_cache(key)
    assert cached is not None
    assert cached["content"] == "saved"


# ── LLMClient: retry on rate limit ───────────────────────────────────────────


async def test_retry_on_rate_limit_succeeds(
    isolated_cache: Path, mock_create: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    mock_create.side_effect = [_rate_limit_error(), _fake_message(text="ok")]

    client = LLMClient()
    result = await client.complete("sys", "usr", "claude-sonnet-4-6")

    assert mock_create.call_count == 2
    assert result["content"] == "ok"


async def test_retry_calls_sleep(
    isolated_cache: Path, mock_create: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)
    mock_create.side_effect = [_rate_limit_error(), _fake_message()]

    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6")

    sleep_mock.assert_called_once()


# ── LLMClient: retry exhausted ───────────────────────────────────────────────


async def test_retry_exhausted_raises(
    isolated_cache: Path, mock_create: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    mock_create.side_effect = _rate_limit_error()

    with pytest.raises(anthropic.RateLimitError):
        await LLMClient().complete("sys", "usr", "claude-sonnet-4-6")


async def test_retry_exhausted_attempts_max_retries(
    isolated_cache: Path, mock_create: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    mock_create.side_effect = _rate_limit_error()

    with pytest.raises(anthropic.RateLimitError):
        await LLMClient().complete("sys", "usr", "claude-sonnet-4-6")

    assert mock_create.call_count == 5  # _MAX_RETRIES


# ── LLMClient: retry-after header ────────────────────────────────────────────


async def test_valid_retry_after_used_as_delay(
    isolated_cache: Path, mock_create: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)
    mock_create.side_effect = [_rate_limit_error(retry_after="2.5"), _fake_message()]

    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6")

    sleep_mock.assert_called_once_with(2.5)


async def test_invalid_retry_after_falls_back_to_backoff(
    isolated_cache: Path, mock_create: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)
    mock_create.side_effect = [_rate_limit_error(retry_after="abc"), _fake_message()]

    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6")

    # attempt=0 → min(1.0 * 2**0, 60) = 1.0
    sleep_mock.assert_called_once_with(1.0)


# ── LLMClient: cost tracking ──────────────────────────────────────────────────


async def test_cost_returned_in_result(isolated_cache: Path, mock_create: AsyncMock) -> None:
    mock_create.return_value = _fake_message(input_tokens=1000, output_tokens=500)
    result = await LLMClient().complete("sys", "usr", "claude-sonnet-4-6")

    assert result["input_tokens"] == 1000
    assert result["output_tokens"] == 500
    # claude-sonnet-4-6: $3/M input, $15/M output
    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
    assert result["cost_usd"] == pytest.approx(expected, rel=1e-6)


async def test_cost_accumulates_on_client(isolated_cache: Path, mock_create: AsyncMock) -> None:
    mock_create.side_effect = [
        _fake_message(input_tokens=100, output_tokens=50),
        _fake_message(input_tokens=200, output_tokens=100),
    ]
    client = LLMClient()
    await client.complete("sys", "q1", "claude-sonnet-4-6")
    await client.complete("sys", "q2", "claude-sonnet-4-6")

    assert client.total_input_tokens == 300
    assert client.total_output_tokens == 150
    assert client.total_cost_usd == pytest.approx(
        (300 * 3.0 + 150 * 15.0) / 1_000_000, rel=1e-6
    )


async def test_unknown_model_uses_default_pricing(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    mock_create.return_value = _fake_message(input_tokens=1_000_000, output_tokens=1_000_000)
    result = await LLMClient().complete("sys", "usr", "claude-unknown-model")

    # default: $3/M input, $15/M output → (3.0 + 15.0) = 18.0 USD
    assert result["cost_usd"] == pytest.approx(18.0, rel=1e-6)


# ── LLMClient: no_cache flag ──────────────────────────────────────────────────


async def test_no_cache_bypasses_cache_read(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    key = get_cache_key("claude-sonnet-4-6", "sys", "usr", 0.0)
    save_to_cache(key, {
        "content": "stale",
        "model": "claude-sonnet-4-6",
        "input_tokens": 1,
        "output_tokens": 1,
        "cost_usd": 0.0,
    })

    mock_create.return_value = _fake_message(text="fresh")
    result = await LLMClient().complete("sys", "usr", "claude-sonnet-4-6", no_cache=True)

    mock_create.assert_called_once()
    assert result["content"] == "fresh"


async def test_no_cache_does_not_write_to_cache(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    mock_create.return_value = _fake_message(text="ephemeral")
    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6", no_cache=True)

    key = get_cache_key("claude-sonnet-4-6", "sys", "usr", 0.0)
    assert load_from_cache(key) is None


# ── LLMClient: tools parameter ───────────────────────────────────────────────


_SAMPLE_TOOLS = [
    {
        "name": "get_stats",
        "description": "Retrieve NBA statistics.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
]


async def test_tools_passed_to_api_create(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6", tools=_SAMPLE_TOOLS)
    call_kwargs = mock_create.call_args.kwargs
    assert "tools" in call_kwargs
    assert call_kwargs["tools"] == _SAMPLE_TOOLS


async def test_no_tools_omits_tools_kwarg(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6")
    call_kwargs = mock_create.call_args.kwargs
    assert "tools" not in call_kwargs


async def test_tools_produces_different_cache_key_than_no_tools(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    # First call without tools — cached.
    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6")
    # Second call with tools — different cache key, so API is called again.
    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6", tools=_SAMPLE_TOOLS)
    assert mock_create.call_count == 2


async def test_tools_result_cached_separately(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6", tools=_SAMPLE_TOOLS)
    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6", tools=_SAMPLE_TOOLS)
    assert mock_create.call_count == 1  # second call is a cache hit


async def test_different_tools_produce_different_cache_keys(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    tools_a = [{"name": "tool_a", "input_schema": {"type": "object"}}]
    tools_b = [{"name": "tool_b", "input_schema": {"type": "object"}}]
    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6", tools=tools_a)
    await LLMClient().complete("sys", "usr", "claude-sonnet-4-6", tools=tools_b)
    assert mock_create.call_count == 2


# ── LLMClient: tool_use blocks in response ───────────────────────────────────


async def test_tool_use_block_exposed_in_tool_calls(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    mock_create.return_value = _fake_tooluse_message(
        tool_name="classify_question",
        tool_input={"category": "factual", "reasoning": "ok"},
    )
    result = await LLMClient().complete(
        "sys", "usr", "claude-sonnet-4-6", tools=_SAMPLE_TOOLS
    )
    assert result["content"] == ""
    assert isinstance(result["tool_calls"], list)
    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    assert call["name"] == "classify_question"
    assert call["input"]["category"] == "factual"
    assert call["id"] == "tool_1"


async def test_text_and_tool_use_both_exposed(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    mock_create.return_value = _fake_tooluse_message(
        tool_name="t",
        tool_input={"x": 1},
        text="preamble",
    )
    result = await LLMClient().complete(
        "sys", "usr", "claude-sonnet-4-6", tools=_SAMPLE_TOOLS
    )
    assert result["content"] == "preamble"
    assert len(result["tool_calls"]) == 1


async def test_text_only_response_has_empty_tool_calls(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    mock_create.return_value = _fake_message(text="hello")
    result = await LLMClient().complete("sys", "usr", "claude-sonnet-4-6")
    assert result["content"] == "hello"
    assert result["tool_calls"] == []


async def test_tool_calls_round_trip_through_cache(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    mock_create.return_value = _fake_tooluse_message(
        tool_name="x",
        tool_input={"a": 1},
    )
    first = await LLMClient().complete(
        "sys", "usr", "claude-sonnet-4-6", tools=_SAMPLE_TOOLS
    )
    second = await LLMClient().complete(
        "sys", "usr", "claude-sonnet-4-6", tools=_SAMPLE_TOOLS
    )
    assert second["cached"] is True
    assert second["tool_calls"] == first["tool_calls"]


async def test_tool_choice_passed_to_api(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    mock_create.return_value = _fake_tooluse_message(
        tool_name="x", tool_input={"a": 1}
    )
    await LLMClient().complete(
        "sys",
        "usr",
        "claude-sonnet-4-6",
        tools=_SAMPLE_TOOLS,
        tool_choice={"type": "tool", "name": "x"},
    )
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "x"}


async def test_tool_choice_affects_cache_key(
    isolated_cache: Path, mock_create: AsyncMock
) -> None:
    mock_create.return_value = _fake_tooluse_message(
        tool_name="x", tool_input={"a": 1}
    )
    await LLMClient().complete(
        "sys", "usr", "claude-sonnet-4-6", tools=_SAMPLE_TOOLS
    )
    await LLMClient().complete(
        "sys",
        "usr",
        "claude-sonnet-4-6",
        tools=_SAMPLE_TOOLS,
        tool_choice={"type": "tool", "name": "x"},
    )
    # Different tool_choice → different cache key → second call hits API again.
    assert mock_create.call_count == 2
