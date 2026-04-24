"""Tests for Milestone 3.5: stats DB schema and SQLAgent."""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.build_stats_db as build_module
from rageval.demo.sql_agent import SQLAgent, _validate_sql
from rageval.types import SQLResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Build a fresh seed database in a temp directory."""
    path = tmp_path / "nba.db"
    build_module.build(db_path=path)
    return path


@pytest.fixture()
def db(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    yield con
    con.close()


def _make_agent(db_path: Path, llm_json: str) -> SQLAgent:
    """Return a SQLAgent whose LLM always replies with *llm_json*."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": llm_json})
    return SQLAgent(llm=llm, db_path=db_path)


# ---------------------------------------------------------------------------
# Schema / seed data tests
# ---------------------------------------------------------------------------


def test_full_name_column_exists(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT full_name FROM players").fetchall()
    assert len(rows) > 0
    names = [r["full_name"] for r in rows]
    assert "Jayson Tatum" in names
    assert "LeBron James" in names


def test_players_schema_columns(db: sqlite3.Connection) -> None:
    info = db.execute("PRAGMA table_info(players)").fetchall()
    cols = {r["name"] for r in info}
    required = {
        "player_id", "full_name", "first_name", "last_name",
        "position", "height_inches", "weight_lbs",
        "birth_date", "draft_year", "draft_pick",
    }
    assert required <= cols


def test_season_id_is_text(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT season_id FROM seasons").fetchall()
    ids = {r["season_id"] for r in rows}
    assert "2022-23" in ids
    assert "2023-24" in ids


def test_player_season_stats_full_column_names(db: sqlite3.Connection) -> None:
    info = db.execute("PRAGMA table_info(player_season_stats)").fetchall()
    cols = {r["name"] for r in info}
    required = {
        "points_per_game", "rebounds_per_game", "assists_per_game",
        "steals_per_game", "blocks_per_game", "turnovers_per_game",
        "minutes_per_game", "true_shooting_pct", "effective_fg_pct",
        "usage_rate", "player_efficiency_rating", "win_shares",
        "box_plus_minus", "vorp",
    }
    assert required <= cols


def test_team_season_stats_full_column_names(db: sqlite3.Connection) -> None:
    info = db.execute("PRAGMA table_info(team_season_stats)").fetchall()
    cols = {r["name"] for r in info}
    required = {
        "points_per_game", "opp_points_per_game",
        "pace", "offensive_rating", "defensive_rating", "net_rating",
    }
    assert required <= cols


def test_teams_schema_columns(db: sqlite3.Connection) -> None:
    info = db.execute("PRAGMA table_info(teams)").fetchall()
    cols = {r["name"] for r in info}
    assert {"team_id", "team_abbr", "team_name", "team_city", "conference", "division"} <= cols


def test_seed_counts(db: sqlite3.Connection) -> None:
    assert db.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 3
    assert db.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 9
    assert db.execute("SELECT COUNT(*) FROM seasons").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM player_season_stats").fetchone()[0] >= 5


# ---------------------------------------------------------------------------
# Direct SQL query tests (validates schema is query-able)
# ---------------------------------------------------------------------------


def test_select_full_name(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT full_name FROM players").fetchall()
    assert len(rows) == 9


def test_leader_query_points_per_game(db: sqlite3.Connection) -> None:
    sql = """
        SELECT p.full_name, s.points_per_game
        FROM player_season_stats s
        JOIN players p ON p.player_id = s.player_id
        WHERE s.season_id = '2023-24'
        ORDER BY s.points_per_game DESC
        LIMIT 5
    """
    rows = db.execute(sql).fetchall()
    assert len(rows) > 0
    # First row should be the highest scorer
    top = rows[0]
    assert top["full_name"] is not None
    assert top["points_per_game"] > 0


# ---------------------------------------------------------------------------
# _validate_sql unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sql", [
    "SELECT * FROM players",
    "select full_name from players",
    (
        "SELECT p.full_name, s.points_per_game FROM player_season_stats s "
        "JOIN players p ON p.player_id = s.player_id"
    ),
])
def test_validate_sql_accepts_select(sql: str) -> None:
    assert _validate_sql(sql) is None


@pytest.mark.parametrize("sql", [
    "DROP TABLE players",
    "INSERT INTO players VALUES (1,'x','x','x',NULL,NULL,NULL,NULL,NULL,NULL)",
    "UPDATE players SET full_name='x'",
    "DELETE FROM players",
    "ALTER TABLE players ADD COLUMN foo TEXT",
    "CREATE TABLE foo (id INTEGER)",
])
def test_validate_sql_rejects_forbidden(sql: str) -> None:
    assert _validate_sql(sql) is not None


def test_validate_sql_rejects_non_select() -> None:
    assert _validate_sql("PRAGMA table_info(players)") is not None


# ---------------------------------------------------------------------------
# SQLAgent integration tests (mocked LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_returns_rows(db_path: Path) -> None:
    sql = "SELECT full_name FROM players ORDER BY full_name"
    agent = _make_agent(db_path, f'{{"sql": "{sql}", "reasoning": "list players"}}')
    result = await agent.generate_and_execute("List all players")
    assert isinstance(result, SQLResult)
    assert result.error is None
    assert len(result.rows) == 9
    assert all("full_name" in row for row in result.rows)


@pytest.mark.asyncio
async def test_agent_leader_query(db_path: Path) -> None:
    sql = (
        "SELECT p.full_name, s.points_per_game "
        "FROM player_season_stats s "
        "JOIN players p ON p.player_id = s.player_id "
        "WHERE s.season_id = '2023-24' "
        "ORDER BY s.points_per_game DESC LIMIT 5"
    )
    agent = _make_agent(db_path, f'{{"sql": "{sql}", "reasoning": "top scorers"}}')
    result = await agent.generate_and_execute("Who led in points per game in 2023-24?")
    assert result.error is None
    assert len(result.rows) > 0
    assert result.rows[0]["points_per_game"] >= result.rows[-1]["points_per_game"]


@pytest.mark.asyncio
async def test_agent_rejects_non_select(db_path: Path) -> None:
    # INSERT is a forbidden keyword AND not a SELECT — either error is valid.
    bad_sql = "INSERT INTO players VALUES (99,'x','x','x',NULL,NULL,NULL,NULL,NULL,NULL)"
    agent = _make_agent(
        db_path,
        f'{{"sql": "{bad_sql}", "reasoning": "bad"}}',
    )
    result = await agent.generate_and_execute("Insert a player")
    assert result.error is not None
    assert "select" in result.error.lower() or "forbidden" in result.error.lower()


@pytest.mark.asyncio
async def test_agent_handles_invalid_json(db_path: Path) -> None:
    agent = _make_agent(db_path, "not json at all")
    result = await agent.generate_and_execute("anything")
    assert result.error is not None
    assert "invalid json" in result.error.lower()


@pytest.mark.asyncio
async def test_agent_empty_result(db_path: Path) -> None:
    sql = "SELECT full_name FROM players WHERE full_name = 'Nonexistent Player'"
    agent = _make_agent(db_path, f'{{"sql": "{sql}", "reasoning": "no match"}}')
    result = await agent.generate_and_execute("Find nonexistent player")
    assert result.error is None
    assert result.rows == []


@pytest.mark.asyncio
async def test_agent_bad_sql_execution(db_path: Path) -> None:
    agent = _make_agent(
        db_path,
        '{"sql": "SELECT nonexistent_col FROM players", "reasoning": "bad col"}',
    )
    result = await agent.generate_and_execute("Bad query")
    assert result.error is not None


@pytest.mark.asyncio
async def test_agent_missing_db(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path / "missing.db",
        '{"sql": "SELECT 1", "reasoning": "simple"}',
    )
    result = await agent.generate_and_execute("anything")
    assert result.error is not None
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_agent_result_is_list_of_dicts(db_path: Path) -> None:
    sql = "SELECT full_name, position FROM players LIMIT 3"
    agent = _make_agent(db_path, f'{{"sql": "{sql}", "reasoning": "sample"}}')
    result = await agent.generate_and_execute("Sample players")
    assert result.error is None
    assert isinstance(result.rows, list)
    for row in result.rows:
        assert isinstance(row, dict)
        assert "full_name" in row
        assert "position" in row
