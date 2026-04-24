"""Tests for Milestone 3.5: stats DB schema and SQLAgent."""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.build_stats_db as build_module
from rageval.demo.sql_agent import _TOOL_NAME, SQLAgent, _validate_sql
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
    """Return a SQLAgent whose LLM always replies with *llm_json* as text (fallback path)."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": llm_json, "tool_calls": []})
    return SQLAgent(llm=llm, db_path=db_path)


def _make_tooluse_agent(db_path: Path, sql: str, reasoning: str = "test") -> SQLAgent:
    """Return a SQLAgent whose LLM replies via a tool_use call."""
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value={
            "content": "",
            "tool_calls": [
                {
                    "id": "t1",
                    "name": _TOOL_NAME,
                    "input": {"reasoning": reasoning, "sql": sql},
                }
            ],
        }
    )
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
# _validate_sql: additional safety edge cases (multi-statement, comments, ATTACH)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sql", [
    "SELECT 1; SELECT 2",
    "SELECT * FROM players; DROP TABLE players",
    "SELECT * FROM players;SELECT * FROM teams",
])
def test_validate_sql_rejects_multi_statement(sql: str) -> None:
    err = _validate_sql(sql)
    assert err is not None
    assert "multiple" in err.lower() or "forbidden" in err.lower()


def test_validate_sql_rejects_attach() -> None:
    assert _validate_sql("ATTACH DATABASE 'foo.db' AS foo") is not None


def test_validate_sql_rejects_detach() -> None:
    assert _validate_sql("DETACH DATABASE foo") is not None


def test_validate_sql_rejects_pragma() -> None:
    assert _validate_sql("PRAGMA table_info(players)") is not None


def test_validate_sql_rejects_comment_hiding_drop() -> None:
    # The "SELECT" prefix here is inside a line comment only.
    sql = "-- SELECT\nDROP TABLE players"
    assert _validate_sql(sql) is not None


def test_validate_sql_rejects_block_comment_hiding_insert() -> None:
    sql = "SELECT * FROM players /* hide */ ; /* */ INSERT INTO players VALUES (1)"
    err = _validate_sql(sql)
    assert err is not None


def test_validate_sql_rejects_empty() -> None:
    assert _validate_sql("") is not None
    assert _validate_sql("   ") is not None


def test_validate_sql_accepts_trailing_semicolon() -> None:
    # A single trailing semicolon is harmless.
    assert _validate_sql("SELECT 1;") is None


def test_validate_sql_accepts_select_with_line_comment() -> None:
    # SELECT with a comment is fine (no forbidden keywords inside the comment).
    sql = "SELECT full_name FROM players -- list all\n"
    assert _validate_sql(sql) is None


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
async def test_agent_handles_no_tool_call(db_path: Path) -> None:
    # Neither a tool_call nor valid JSON text → must not execute anything.
    agent = _make_agent(db_path, "not json at all")
    result = await agent.generate_and_execute("anything")
    assert result.error is not None
    assert "tool call" in result.error.lower() or "sql" in result.error.lower()
    assert result.rows == []


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
async def test_agent_executes_tooluse_response(db_path: Path) -> None:
    agent = _make_tooluse_agent(db_path, "SELECT full_name FROM players")
    result = await agent.generate_and_execute("list players")
    assert result.error is None
    assert len(result.rows) == 9


@pytest.mark.asyncio
async def test_agent_rejects_tooluse_multi_statement(db_path: Path) -> None:
    agent = _make_tooluse_agent(db_path, "SELECT 1; SELECT 2")
    result = await agent.generate_and_execute("multi")
    assert result.error is not None
    assert "multiple" in result.error.lower() or "forbidden" in result.error.lower()


@pytest.mark.asyncio
async def test_agent_rejects_tooluse_attach(db_path: Path) -> None:
    agent = _make_tooluse_agent(db_path, "ATTACH DATABASE 'foo.db' AS foo")
    result = await agent.generate_and_execute("attach")
    assert result.error is not None
    assert result.rows == []


@pytest.mark.asyncio
async def test_agent_passes_tool_schema(db_path: Path) -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value={
            "content": "",
            "tool_calls": [
                {"id": "t", "name": _TOOL_NAME, "input": {"reasoning": "x", "sql": "SELECT 1"}}
            ],
        }
    )
    agent = SQLAgent(llm=llm, db_path=db_path)
    await agent.generate_and_execute("anything")
    kwargs = llm.complete.call_args.kwargs
    tools = kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == _TOOL_NAME
    assert "sql" in tools[0]["input_schema"]["properties"]
    assert kwargs["tool_choice"] == {"type": "tool", "name": _TOOL_NAME}


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


# ---------------------------------------------------------------------------
# Prompt file loading (Fix: sql_agent must load from prompts/sql_agent/v1.txt)
# ---------------------------------------------------------------------------


def test_prompt_file_exists() -> None:
    from rageval.demo.sql_agent import _PROMPT_PATH
    assert _PROMPT_PATH.exists(), f"Prompt file missing: {_PROMPT_PATH}"


def test_agent_system_prompt_matches_prompt_file(db_path: Path) -> None:
    from rageval.demo.sql_agent import _PROMPT_PATH, SQLAgent
    agent = SQLAgent(db_path=db_path)
    assert agent._system == _PROMPT_PATH.read_text(encoding="utf-8")


def test_prompt_file_contains_select_rule() -> None:
    from rageval.demo.sql_agent import _PROMPT_PATH
    content = _PROMPT_PATH.read_text(encoding="utf-8")
    assert "SELECT" in content


# ---------------------------------------------------------------------------
# ingestion_log table (Fix: add ingestion_log per PROJECT_PLAN.md)
# ---------------------------------------------------------------------------


def test_ingestion_log_table_exists(db: sqlite3.Connection) -> None:
    info = db.execute("PRAGMA table_info(ingestion_log)").fetchall()
    assert len(info) > 0, "ingestion_log table not found"


def test_ingestion_log_columns(db: sqlite3.Connection) -> None:
    info = db.execute("PRAGMA table_info(ingestion_log)").fetchall()
    cols = {r["name"] for r in info}
    assert {"id", "run_at", "source", "records_added", "notes"} <= cols


def test_ingestion_log_has_one_seed_row(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT * FROM ingestion_log").fetchall()
    assert len(rows) == 1


def test_ingestion_log_seed_row_fields(db: sqlite3.Connection) -> None:
    row = db.execute("SELECT * FROM ingestion_log").fetchone()
    assert row["source"] == "seed"
    assert row["records_added"] > 0
    assert row["run_at"] is not None


# ---------------------------------------------------------------------------
# Article / chunk placeholder tables (empty, created by schema init)
# ---------------------------------------------------------------------------


def test_articles_table_exists(db: sqlite3.Connection) -> None:
    info = db.execute("PRAGMA table_info(articles)").fetchall()
    cols = {r["name"] for r in info}
    required = {
        "article_id", "title", "source", "author", "url",
        "publish_date", "full_text", "word_count", "ingested_at",
    }
    assert required <= cols


def test_article_chunks_table_exists(db: sqlite3.Connection) -> None:
    info = db.execute("PRAGMA table_info(article_chunks)").fetchall()
    cols = {r["name"] for r in info}
    assert {"chunk_id", "article_id", "chunk_index", "content", "token_count"} <= cols


def test_article_tables_are_empty_on_seed_build(db: sqlite3.Connection) -> None:
    # Corpus ingestion is Milestone 7 — the seed build must not populate these.
    assert db.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM article_chunks").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Real-mode ingestion (mocked nba_api) — verifies the code path without network.
# ---------------------------------------------------------------------------


def test_build_real_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise build_real with mocked nba_api endpoints + static teams."""
    import sys

    # Stub nba_api modules *before* build_real imports them.
    fake_static_teams = MagicMock()
    fake_static_teams.get_teams = lambda: [
        {
            "id": 1610612738,
            "abbreviation": "BOS",
            "full_name": "Boston Celtics",
            "city": "Boston",
            "nickname": "Celtics",
        }
    ]

    fake_player_stats = MagicMock()

    def _player_stats_cls(season: str) -> MagicMock:  # noqa: ARG001
        inst = MagicMock()
        inst.get_dict.return_value = {
            "resultSets": [
                {
                    "headers": [
                        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
                        "GP", "GS", "MIN", "PTS", "REB", "AST",
                        "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT",
                    ],
                    "rowSet": [
                        [
                            1,
                            "Test Player",
                            1610612738,
                            70, 70, 34.0, 25.0, 8.0, 5.0,
                            1.1, 0.5, 2.5, 0.5, 0.38, 0.85,
                        ]
                    ],
                }
            ]
        }
        return inst

    fake_player_stats.LeagueDashPlayerStats = _player_stats_cls

    fake_team_stats = MagicMock()

    def _team_stats_cls(season: str) -> MagicMock:  # noqa: ARG001
        inst = MagicMock()
        inst.get_dict.return_value = {
            "resultSets": [
                {
                    "headers": ["TEAM_ID", "W", "L", "PTS"],
                    "rowSet": [[1610612738, 50, 32, 118.0]],
                }
            ]
        }
        return inst

    fake_team_stats.LeagueDashTeamStats = _team_stats_cls

    # Build the fake nba_api package tree.
    nba_api_mod = MagicMock()
    stats_mod = MagicMock()
    endpoints_mod = MagicMock()
    static_mod = MagicMock()
    endpoints_mod.leaguedashplayerstats = fake_player_stats
    endpoints_mod.leaguedashteamstats = fake_team_stats
    static_mod.teams = fake_static_teams
    stats_mod.endpoints = endpoints_mod
    stats_mod.static = static_mod
    nba_api_mod.stats = stats_mod

    monkeypatch.setitem(sys.modules, "nba_api", nba_api_mod)
    monkeypatch.setitem(sys.modules, "nba_api.stats", stats_mod)
    monkeypatch.setitem(sys.modules, "nba_api.stats.endpoints", endpoints_mod)
    monkeypatch.setitem(sys.modules, "nba_api.stats.static", static_mod)
    monkeypatch.setitem(
        sys.modules, "nba_api.stats.endpoints.leaguedashplayerstats", fake_player_stats
    )
    monkeypatch.setitem(
        sys.modules, "nba_api.stats.endpoints.leaguedashteamstats", fake_team_stats
    )
    monkeypatch.setitem(sys.modules, "nba_api.stats.static.teams", fake_static_teams)

    db_path = tmp_path / "nba.db"
    raw_dir = tmp_path / "raw"
    counts = build_module.build_real(
        db_path=db_path, seasons=["2023-24"], raw_dir=raw_dir
    )

    assert counts["teams"] == 1
    assert counts["players"] == 1
    assert counts["player_season_stats"] == 1
    assert counts["team_season_stats"] == 1

    # Raw API responses saved.
    assert (raw_dir / "teams.json").exists()
    assert (raw_dir / "player_stats_2023-24.json").exists()
    assert (raw_dir / "team_stats_2023-24.json").exists()

    # Ingestion log has an nba_api row.
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT * FROM ingestion_log WHERE source = 'nba_api'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["records_added"] > 0

        # Data actually landed.
        assert con.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 1
        player = con.execute(
            "SELECT full_name, first_name, last_name FROM players"
        ).fetchone()
        assert player["full_name"] == "Test Player"
        pss = con.execute(
            "SELECT points_per_game, games_played FROM player_season_stats"
        ).fetchone()
        assert pss["points_per_game"] == 25.0
        assert pss["games_played"] == 70
    finally:
        con.close()


def test_build_real_retries_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_with_retries should retry a failing call then succeed."""
    import time as time_module

    monkeypatch.setattr(time_module, "sleep", lambda _s: None)

    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("flaky")
        return "ok"

    result = build_module._with_retries(fn, attempts=3, base_delay=0.0, label="t")
    assert result == "ok"
    assert calls["n"] == 2
