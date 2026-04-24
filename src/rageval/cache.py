import hashlib
import json
from pathlib import Path
from typing import Any, cast

_CACHE_DIR = Path(".rageval_cache")


def get_cache_key(
    model: str,
    system: str,
    user: str,
    temperature: float,
    tool_schema: dict[str, Any] | None = None,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "user": user,
            "temperature": temperature,
            "tool_schema": tool_schema,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / key[:2] / f"{key}.json"


def load_from_cache(key: str) -> dict[str, Any] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return cast(dict[str, Any], raw)
        return None
    except (json.JSONDecodeError, OSError):
        return None


def save_to_cache(key: str, response_dict: dict[str, Any]) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response_dict, indent=2), encoding="utf-8")
