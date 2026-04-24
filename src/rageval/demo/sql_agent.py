"""SQL agent: generates and executes SQL from natural language questions."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from rageval.llm_client import LLMClient
from rageval.types import SQLResult

_DB_PATH = Path(__file__).parents[4] / "data" / "nba.db"

_MODEL = "claude-haiku-4-5-20251001"

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)

_SCHEMA_DDL = """
teams(team_id, team_abbr, team_name, team_city, conference, division)

players(player_id, full_name, first_name, last_name, position,
        height_inches, weight_lbs, birth_date, draft_year, draft_pick)

seasons(season_id TEXT, start_year, end_year)
  -- season_id examples: '2022-23', '2023-24'

games(game_id, game_date, season_id, home_team_id, away_team_id,
      home_score, away_score, game_type)
  -- game_type: 'regular' | 'playoff'

player_game_stats(id, game_id, player_id, team_id, minutes, points,
                  rebounds, assists, steals, blocks, turnovers,
                  fg_made, fg_attempted, fg3_made, fg3_attempted,
                  ft_made, ft_attempted, plus_minus)

player_season_stats(id, player_id, season_id, team_id,
                    games_played, games_started, minutes_per_game,
                    points_per_game, rebounds_per_game, assists_per_game,
                    steals_per_game, blocks_per_game, turnovers_per_game,
                    fg_pct, fg3_pct, ft_pct,
                    true_shooting_pct, effective_fg_pct, usage_rate,
                    player_efficiency_rating, win_shares, box_plus_minus, vorp)

team_season_stats(id, team_id, season_id, wins, losses,
                  points_per_game, opp_points_per_game,
                  pace, offensive_rating, defensive_rating, net_rating)

Foreign keys:
  players has no team FK (players move; link via player_season_stats.team_id)
  games.season_id -> seasons.season_id
  games.home_team_id / away_team_id -> teams.team_id
  player_game_stats.game_id -> games.game_id
  player_game_stats.player_id -> players.player_id
  player_game_stats.team_id -> teams.team_id
  player_season_stats.player_id -> players.player_id
  player_season_stats.season_id -> seasons.season_id
  player_season_stats.team_id -> teams.team_id
  team_season_stats.team_id -> teams.team_id
  team_season_stats.season_id -> seasons.season_id
""".strip()

_SYSTEM_PROMPT = f"""You are an NBA statistics SQL assistant.

Database schema:
{_SCHEMA_DDL}

Given a natural-language question, respond with ONLY valid JSON in this exact format:
{{"sql": "<SELECT query>", "reasoning": "<brief explanation>"}}

Rules:
- The sql field must be a single SELECT statement.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or any DDL/DML.
- Use players.full_name when showing player names.
- Use descriptive column aliases (e.g. points_per_game, not ppg).
- If the question cannot be answered from the schema return:
  {{"sql": "SELECT 1", "reasoning": "Cannot answer: <reason>"}}
- Output only the JSON object — no markdown, no code fences, no extra text.
"""


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

    async def generate_and_execute(self, question: str) -> SQLResult:
        llm_response = await self._llm.complete(
            system=_SYSTEM_PROMPT,
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
