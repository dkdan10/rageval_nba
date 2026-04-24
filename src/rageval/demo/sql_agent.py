"""SQL agent: generates and executes SQL from natural language questions.

Uses Anthropic tool-use for structured output. The agent validates generated SQL
against a strict allowlist before execution (SELECT-only, single statement,
no DDL/DML/PRAGMA/ATTACH/DETACH, no comments that could hide forbidden statements).
"""

import re
import sqlite3
from pathlib import Path
from typing import Any

from rageval.llm_client import LLMClient
from rageval.types import SQLResult

_DB_PATH = Path(__file__).parents[3] / "data" / "nba.db"
_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "sql_agent" / "v1.txt"
_MODEL = "claude-haiku-4-5-20251001"
_TOOL_NAME = "run_sql"
_MAX_ROWS = 500

_FORBIDDEN = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|"
    r"GRANT|REVOKE|VACUUM|ANALYZE|REINDEX|PRAGMA|ATTACH|DETACH"
    r")\b",
    re.IGNORECASE,
)

_SQL_AGENT_TOOL: dict[str, Any] = {
    "name": _TOOL_NAME,
    "description": (
        "Submit a single SELECT query against the NBA stats database. "
        "The query is validated and executed read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "One short sentence describing the query strategy.",
            },
            "sql": {
                "type": "string",
                "description": "A single SELECT statement. No trailing semicolon.",
            },
        },
        "required": ["reasoning", "sql"],
    },
}


def _strip_sql_comments(sql: str) -> str:
    """Remove `--` line comments and `/* ... */` block comments from *sql*."""
    without_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", without_block)


def _validate_sql(sql: str) -> str | None:
    """Return an error string if *sql* is unsafe; None if valid.

    Enforces: single statement, SELECT-only, no forbidden keywords (after
    comment stripping), no PRAGMA/ATTACH/DETACH.
    """
    stripped = sql.strip()
    if not stripped:
        return "Generated SQL is empty"

    # Reject semicolon-chained multi-statements. A trailing semicolon is ok,
    # but no statement content may follow it.
    decommented = _strip_sql_comments(stripped).strip()
    if ";" in decommented.rstrip(";") and decommented.rstrip(";").count(";") > 0:
        return "Generated SQL contains multiple statements (';' not allowed)"

    # Detect forbidden keywords even when hidden by comments: check the
    # comment-stripped form.
    if _FORBIDDEN.search(decommented):
        return "Generated SQL contains a forbidden keyword"

    # Must begin with SELECT (or WITH ... SELECT, but keep it simple —
    # only plain SELECT is accepted here).
    if not decommented.upper().lstrip("(").startswith("SELECT"):
        return "Generated SQL is not a SELECT statement"

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
            tools=[_SQL_AGENT_TOOL],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
        )

        sql = _extract_sql(llm_response)
        if sql is None:
            content = str(llm_response.get("content", ""))
            return SQLResult(
                query="",
                rows=[],
                error=f"LLM did not produce a tool call with SQL: {content[:200]}",
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
                rows: list[dict[str, Any]] = [
                    dict(row) for row in cursor.fetchmany(_MAX_ROWS + 1)
                ]
            finally:
                con.close()
        except sqlite3.Error as exc:
            return SQLResult(query=sql, rows=[], error=str(exc))

        if len(rows) > _MAX_ROWS:
            return SQLResult(
                query=sql,
                rows=rows[:_MAX_ROWS],
                error=f"Query returned more than {_MAX_ROWS} rows; truncated.",
            )
        return SQLResult(query=sql, rows=rows)


def _extract_sql(response: dict[str, Any]) -> str | None:
    """Pull the SQL string from the tool-use response, or fall back to JSON text."""
    tool_calls: list[dict[str, Any]] = response.get("tool_calls") or []
    for call in tool_calls:
        if call.get("name") != _TOOL_NAME:
            continue
        inp = call.get("input")
        if isinstance(inp, dict) and isinstance(inp.get("sql"), str):
            return str(inp["sql"])

    # Backwards-compatible fallback: some tests/mocks return raw JSON text in `content`.
    content = response.get("content")
    if isinstance(content, str) and content.strip():
        import json

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("sql"), str):
            return str(parsed["sql"])
    return None
