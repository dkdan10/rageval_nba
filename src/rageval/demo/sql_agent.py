"""SQL agent: generates and executes SQL from natural language questions."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from rageval.llm_client import LLMClient
from rageval.types import SQLResult

_DB_PATH = Path(__file__).parents[3] / "data" / "nba.db"
_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "sql_agent" / "v1.txt"
_MODEL = "claude-haiku-4-5-20251001"

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)


def _validate_sql(sql: str) -> str | None:
    """Return an error string if sql is unsafe or not a SELECT; None if valid."""
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return "Generated SQL is not a SELECT statement"
    if _FORBIDDEN.search(stripped):
        return "Generated SQL contains a forbidden keyword"
    return None


class SQLAgent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        db_path: Path | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._db_path = db_path or _DB_PATH
        self._system: str = _PROMPT_PATH.read_text(encoding="utf-8")

    async def generate_and_execute(self, question: str) -> SQLResult:
        llm_response = await self._llm.complete(
            system=self._system,
            user=question,
            model=_MODEL,
            temperature=0.0,
        )

        raw: str = llm_response["content"]

        try:
            parsed: dict[str, Any] = json.loads(raw)
            sql: str = str(parsed["sql"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return SQLResult(
                query="",
                rows=[],
                error=f"LLM returned invalid JSON: {raw[:200]}",
            )

        error = _validate_sql(sql)
        if error is not None:
            return SQLResult(query=sql, rows=[], error=error)

        return self._execute(sql)

    def _execute(self, sql: str) -> SQLResult:
        if not self._db_path.exists():
            return SQLResult(
                query=sql,
                rows=[],
                error=(
                    f"Database not found: {self._db_path}. "
                    "Run scripts/build_stats_db.py first."
                ),
            )

        try:
            con = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            try:
                cursor = con.execute(sql)
                rows: list[dict[str, Any]] = [dict(row) for row in cursor.fetchall()]
            finally:
                con.close()
        except sqlite3.Error as exc:
            return SQLResult(query=sql, rows=[], error=str(exc))

        return SQLResult(query=sql, rows=rows)
