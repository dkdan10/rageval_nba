"""Small helpers for loading sqlite-vec only when vector search is needed."""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from typing import cast


def load_sqlite_vec(con: sqlite3.Connection) -> tuple[bool, str | None]:
    """Load sqlite-vec into *con*.

    Returns ``(True, None)`` when the extension is available. Returns
    ``(False, reason)`` instead of raising so callers can keep deterministic
    lexical/offline paths working on platforms where sqlite-vec is unavailable.
    """
    try:
        import sqlite_vec  # type: ignore[import-untyped]
    except ImportError as exc:
        return False, f"sqlite_vec import failed: {exc}"

    try:
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
    except sqlite3.Error as exc:
        return False, f"sqlite_vec load failed: {exc}"
    finally:
        with suppress(sqlite3.Error):
            con.enable_load_extension(False)

    return True, None


def serialize_float32(values: list[float]) -> bytes:
    """Serialize a Python float list for sqlite-vec MATCH queries/inserts."""
    import sqlite_vec

    return cast(bytes, sqlite_vec.serialize_float32(values))
