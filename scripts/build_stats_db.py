"""Build the NBA stats SQLite database.

Two modes:

* ``seed`` (default) — fast, deterministic, offline. Inserts a small hand-curated
  fixture useful for tests and demos. Used automatically by pytest.
* ``real`` — calls nba_api to pull real data for 2020-21 through 2024-25,
  with retries and idempotent inserts. Saves raw API responses to ``data/raw/``
  for reproducibility. Logs each run to ``ingestion_log``.

Run:
    uv run python scripts/build_stats_db.py                # seed mode
    uv run python scripts/build_stats_db.py --mode real    # real ingestion

Article/vector tables (``articles``, ``article_chunks``, ``chunk_embeddings``)
are created as empty placeholders. Corpus ingestion is a Milestone 7 task.
``chunk_embeddings`` requires the ``sqlite-vec`` extension; if the extension is
not available the embeddings virtual table is skipped (other code paths should
guard on its existence).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "nba.db"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
REAL_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS teams (
    team_id    INTEGER PRIMARY KEY,
    team_abbr  TEXT NOT NULL UNIQUE,
    team_name  TEXT NOT NULL,
    team_city  TEXT NOT NULL,
    conference TEXT NOT NULL CHECK(conference IN ('East', 'West')),
    division   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS players (
    player_id     INTEGER PRIMARY KEY,
    full_name     TEXT NOT NULL,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    position      TEXT,
    height_inches INTEGER,
    weight_lbs    INTEGER,
    birth_date    TEXT,
    draft_year    INTEGER,
    draft_pick    INTEGER
);

CREATE TABLE IF NOT EXISTS seasons (
    season_id  TEXT PRIMARY KEY,
    start_year INTEGER NOT NULL,
    end_year   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    game_id      INTEGER PRIMARY KEY,
    game_date    TEXT NOT NULL,
    season_id    TEXT NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_score   INTEGER NOT NULL,
    away_score   INTEGER NOT NULL,
    game_type    TEXT NOT NULL CHECK(game_type IN ('regular', 'playoff')),
    FOREIGN KEY (season_id)    REFERENCES seasons(season_id),
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season_id);
CREATE INDEX IF NOT EXISTS idx_games_date   ON games(game_date);

CREATE TABLE IF NOT EXISTS player_game_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id      INTEGER NOT NULL,
    player_id    INTEGER NOT NULL,
    team_id      INTEGER NOT NULL,
    minutes      REAL,
    points       INTEGER,
    rebounds     INTEGER,
    assists      INTEGER,
    steals       INTEGER,
    blocks       INTEGER,
    turnovers    INTEGER,
    fg_made      INTEGER,
    fg_attempted INTEGER,
    fg3_made     INTEGER,
    fg3_attempted INTEGER,
    ft_made      INTEGER,
    ft_attempted INTEGER,
    plus_minus   INTEGER,
    FOREIGN KEY (game_id)   REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id)   REFERENCES teams(team_id)
);
CREATE INDEX IF NOT EXISTS idx_pgs_game   ON player_game_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_pgs_player ON player_game_stats(player_id);

CREATE TABLE IF NOT EXISTS player_season_stats (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id                INTEGER NOT NULL,
    season_id                TEXT NOT NULL,
    team_id                  INTEGER NOT NULL,
    games_played             INTEGER,
    games_started            INTEGER,
    minutes_per_game         REAL,
    points_per_game          REAL,
    rebounds_per_game        REAL,
    assists_per_game         REAL,
    steals_per_game          REAL,
    blocks_per_game          REAL,
    turnovers_per_game       REAL,
    fg_pct                   REAL,
    fg3_pct                  REAL,
    ft_pct                   REAL,
    true_shooting_pct        REAL,
    effective_fg_pct         REAL,
    usage_rate               REAL,
    player_efficiency_rating REAL,
    win_shares               REAL,
    box_plus_minus           REAL,
    vorp                     REAL,
    UNIQUE(player_id, season_id, team_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (season_id) REFERENCES seasons(season_id),
    FOREIGN KEY (team_id)   REFERENCES teams(team_id)
);
CREATE INDEX IF NOT EXISTS idx_pss_season ON player_season_stats(season_id);
CREATE INDEX IF NOT EXISTS idx_pss_player ON player_season_stats(player_id);

CREATE TABLE IF NOT EXISTS team_season_stats (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id           INTEGER NOT NULL,
    season_id         TEXT NOT NULL,
    wins              INTEGER,
    losses            INTEGER,
    points_per_game   REAL,
    opp_points_per_game REAL,
    pace              REAL,
    offensive_rating  REAL,
    defensive_rating  REAL,
    net_rating        REAL,
    UNIQUE(team_id, season_id),
    FOREIGN KEY (team_id)   REFERENCES teams(team_id),
    FOREIGN KEY (season_id) REFERENCES seasons(season_id)
);

-- Placeholder tables for the unstructured (RAG) path. Populated in Milestone 7.
CREATE TABLE IF NOT EXISTS articles (
    article_id   TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source       TEXT NOT NULL,
    author       TEXT,
    url          TEXT NOT NULL,
    publish_date TEXT,
    full_text    TEXT NOT NULL,
    word_count   INTEGER NOT NULL,
    ingested_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS article_chunks (
    chunk_id    TEXT PRIMARY KEY,
    article_id  TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    FOREIGN KEY (article_id) REFERENCES articles(article_id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_article ON article_chunks(article_id);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at        TEXT NOT NULL,
    source        TEXT NOT NULL,
    records_added INTEGER NOT NULL,
    notes         TEXT
);
"""


def _try_create_vector_table(con: sqlite3.Connection) -> bool:
    """Create chunk_embeddings via sqlite-vec if available. Return True on success."""
    try:
        import sqlite_vec  # type: ignore[import-untyped]
    except ImportError:
        return False
    try:
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings "
            "USING vec0(chunk_id TEXT PRIMARY KEY, embedding FLOAT[1024])"
        )
    except sqlite3.Error:
        return False
    return True


# ---------------------------------------------------------------------------
# Seed data (fast, offline; used by tests)
# ---------------------------------------------------------------------------

# (team_id, team_abbr, team_name, team_city, conference, division)
TEAMS = [
    (1, "BOS", "Boston Celtics",        "Boston",        "East", "Atlantic"),
    (2, "GSW", "Golden State Warriors", "San Francisco",  "West", "Pacific"),
    (3, "LAL", "Los Angeles Lakers",    "Los Angeles",   "West", "Pacific"),
]

# (player_id, full_name, first_name, last_name, position,
#  height_inches, weight_lbs, birth_date, draft_year, draft_pick)
PLAYERS = [
    (1, "Jayson Tatum",   "Jayson",  "Tatum",    "F",   80, 210, "1998-03-03", 2017, 3),
    (2, "Jaylen Brown",   "Jaylen",  "Brown",    "G-F", 78, 223, "1996-10-24", 2016, 3),
    (3, "Al Horford",     "Al",      "Horford",  "C",   81, 240, "1986-06-03", 2007, 3),
    (4, "Stephen Curry",  "Stephen", "Curry",    "G",   74, 185, "1988-03-14", 2009, 7),
    (5, "Klay Thompson",  "Klay",    "Thompson", "G",   79, 215, "1990-02-08", 2011, 11),
    (6, "Draymond Green", "Draymond","Green",    "F",   79, 230, "1990-03-04", 2012, 35),
    (7, "LeBron James",   "LeBron",  "James",    "F",   81, 250, "1984-12-30", 2003, 1),
    (8, "Anthony Davis",  "Anthony", "Davis",    "C",   82, 253, "1993-03-11", 2012, 1),
    (9, "Austin Reaves",  "Austin",  "Reaves",   "G",   77, 197, "1998-05-29", None, None),
]

SEASONS = [
    ("2022-23", 2022, 2023),
    ("2023-24", 2023, 2024),
]

GAMES = [
    (1, "2023-01-15", "2022-23", 1, 2, 118, 110, "regular"),
    (2, "2023-02-20", "2022-23", 3, 1, 112, 120, "regular"),
    (3, "2024-01-10", "2023-24", 2, 3, 125, 119, "regular"),
    (4, "2024-03-05", "2023-24", 1, 3, 107,  98, "regular"),
]

PLAYER_GAME_STATS = [
    (1, 1, 1, 36.0, 32, 8,  5, 2, 0, 3, 12, 24, 3, 8, 5,  6,  8),
    (1, 2, 1, 34.0, 24, 6,  4, 1, 0, 2,  9, 18, 2, 6, 4,  5,  6),
    (1, 4, 2, 38.0, 28, 4,  7, 2, 0, 4, 10, 20, 5, 11,3,  3, -8),
    (1, 5, 2, 30.0, 18, 3,  3, 1, 0, 1,  6, 14, 3,  8, 3,  4, -5),
    (2, 7, 3, 37.0, 30, 7,  8, 1, 1, 2, 11, 22, 2,  6, 6,  8,  8),
    (2, 8, 3, 34.0, 22, 12, 2, 1, 3, 1,  9, 16, 0,  1, 4,  5,  6),
    (2, 1, 1, 38.0, 36, 7,  4, 2, 0, 4, 13, 25, 4,  9, 6,  7, -8),
    (3, 4, 2, 40.0, 38, 5,  9, 3, 0, 5, 14, 26, 6, 13, 4,  4,  6),
    (3, 7, 3, 36.0, 27, 8,  6, 2, 2, 3, 10, 20, 2,  6, 5,  7, -6),
    (4, 1, 1, 40.0, 41, 9,  6, 1, 0, 2, 15, 27, 5, 11, 6,  7,  9),
    (4, 7, 3, 36.0, 22, 7,  7, 0, 1, 4,  8, 17, 2,  5, 4,  5, -9),
]


def _pss(
    pid: int, sid: str, tid: int,
    gp: int, gs: int, mpg: float, ppg: float, rpg: float, apg: float,
    spg: float, bpg: float, tpg: float,
    fg: float, fg3: float, ft: float,
    ts: float, efg: float, usg: float, per: float, ws: float, bpm: float, vorp: float,
) -> tuple[int, str, int, int, int, float, float, float, float,
           float, float, float, float, float, float,
           float, float, float, float, float, float, float]:
    return (pid, sid, tid, gp, gs, mpg, ppg, rpg, apg,
            spg, bpg, tpg, fg, fg3, ft, ts, efg, usg, per, ws, bpm, vorp)


PLAYER_SEASON_STATS = [
    _pss(1, "2022-23", 1, 74, 74, 36.9, 30.1, 8.8, 4.6, 1.1, 0.7, 3.0, 0.466, 0.351, 0.855,
         0.599, 0.537, 33.2, 26.9, 9.4, 6.2, 3.9),
    _pss(2, "2022-23", 1, 67, 67, 35.9, 26.6, 6.9, 3.5, 1.1, 0.3, 2.7, 0.491, 0.354, 0.711,
         0.587, 0.562, 31.5, 21.2, 7.8, 4.3, 2.6),
    _pss(3, "2022-23", 1, 68, 68, 28.5, 10.2, 7.3, 2.7, 0.9, 1.1, 1.3, 0.447, 0.370, 0.844,
         0.568, 0.510, 14.0, 14.2, 6.8, 2.9, 1.7),
    _pss(4, "2022-23", 2, 56, 56, 34.5, 29.4, 6.1, 6.3, 0.9, 0.4, 3.4, 0.490, 0.427, 0.915,
         0.660, 0.592, 28.5, 25.6, 6.8, 8.2, 4.1),
    _pss(5, "2022-23", 2, 69, 69, 33.0, 21.9, 3.3, 2.4, 0.8, 0.5, 2.0, 0.427, 0.381, 0.837,
         0.561, 0.490, 24.0, 15.7, 3.8, 2.8, 1.2),
    _pss(6, "2022-23", 2, 73, 73, 30.1,  8.5, 7.0, 7.2, 1.0, 0.8, 3.7, 0.445, 0.296, 0.646,
         0.519, 0.478, 17.0, 15.8, 7.7, 5.5, 3.5),
    _pss(7, "2022-23", 3, 55, 55, 35.5, 28.9, 8.3, 6.8, 1.6, 0.6, 3.5, 0.500, 0.324, 0.768,
         0.604, 0.554, 31.0, 24.1, 7.1, 6.3, 3.4),
    _pss(8, "2022-23", 3, 56, 56, 35.5, 25.9, 12.5, 2.6, 1.1, 2.0, 2.1, 0.560, 0.221, 0.782,
         0.622, 0.589, 29.5, 25.3, 7.4, 6.1, 3.3),
    _pss(9, "2022-23", 3, 68, 30, 27.5, 13.0, 4.0, 3.0, 0.8, 0.2, 1.7, 0.510, 0.402, 0.820,
         0.618, 0.566, 19.0, 15.5, 4.9, 3.2, 1.4),
    _pss(1, "2023-24", 1, 74, 74, 35.8, 26.9, 8.1, 4.9, 1.2, 0.6, 2.9, 0.471, 0.376, 0.831,
         0.604, 0.553, 30.5, 25.0, 9.8, 6.9, 4.4),
    _pss(2, "2023-24", 1, 70, 70, 33.5, 23.0, 5.5, 3.6, 1.2, 0.5, 2.5, 0.478, 0.357, 0.702,
         0.571, 0.537, 28.0, 18.7, 6.5, 3.8, 2.2),
    _pss(4, "2023-24", 2, 74, 74, 33.1, 26.4, 4.4, 5.1, 0.7, 0.4, 3.3, 0.453, 0.408, 0.924,
         0.641, 0.545, 27.2, 24.2, 6.8, 7.7, 4.0),
    _pss(7, "2023-24", 3, 71, 71, 35.3, 25.7, 7.3, 8.3, 1.3, 0.5, 3.5, 0.540, 0.410, 0.750,
         0.640, 0.594, 31.4, 24.6, 7.0, 6.5, 3.5),
    _pss(8, "2023-24", 3, 76, 76, 35.2, 24.7, 12.6, 3.5, 1.2, 2.3, 2.2, 0.558, 0.175, 0.798,
         0.621, 0.578, 28.8, 26.9, 8.4, 6.5, 3.8),
]

TEAM_SEASON_STATS = [
    (1, "2022-23", 57, 25, 117.9, 112.5, 98.3, 116.3, 110.9,  5.4),
    (2, "2022-23", 44, 38, 120.2, 115.6, 99.8, 118.3, 113.9,  4.4),
    (3, "2022-23", 43, 39, 117.2, 116.4, 99.0, 115.5, 114.8,  0.7),
    (1, "2023-24", 64, 18, 120.6, 110.5, 99.1, 122.2, 112.0, 10.2),
    (2, "2023-24", 46, 36, 118.4, 115.7, 99.4, 117.6, 115.0,  2.6),
    (3, "2023-24", 47, 35, 118.3, 116.2, 99.7, 116.8, 114.7,  2.1),
]

_SEED_RECORD_COUNT = (
    len(TEAMS)
    + len(PLAYERS)
    + len(SEASONS)
    + len(GAMES)
    + len(PLAYER_GAME_STATS)
    + len(PLAYER_SEASON_STATS)
    + len(TEAM_SEASON_STATS)
)


# ---------------------------------------------------------------------------
# Seed build
# ---------------------------------------------------------------------------


def _init_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    _try_create_vector_table(con)  # best-effort


def build(db_path: Path = DB_PATH) -> None:
    """Build a fresh seed database at *db_path*. Overwrites any existing file."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    try:
        _init_schema(con)

        con.executemany("INSERT INTO teams VALUES (?,?,?,?,?,?)", TEAMS)
        con.executemany("INSERT INTO players VALUES (?,?,?,?,?,?,?,?,?,?)", PLAYERS)
        con.executemany("INSERT INTO seasons VALUES (?,?,?)", SEASONS)
        con.executemany("INSERT INTO games VALUES (?,?,?,?,?,?,?,?)", GAMES)
        con.executemany(
            "INSERT INTO player_game_stats"
            "(game_id,player_id,team_id,minutes,points,rebounds,assists,"
            "steals,blocks,turnovers,fg_made,fg_attempted,fg3_made,"
            "fg3_attempted,ft_made,ft_attempted,plus_minus)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            PLAYER_GAME_STATS,
        )
        con.executemany(
            "INSERT INTO player_season_stats"
            "(player_id,season_id,team_id,games_played,games_started,"
            "minutes_per_game,points_per_game,rebounds_per_game,assists_per_game,"
            "steals_per_game,blocks_per_game,turnovers_per_game,"
            "fg_pct,fg3_pct,ft_pct,true_shooting_pct,effective_fg_pct,"
            "usage_rate,player_efficiency_rating,win_shares,box_plus_minus,vorp)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            PLAYER_SEASON_STATS,
        )
        con.executemany(
            "INSERT INTO team_season_stats"
            "(team_id,season_id,wins,losses,points_per_game,opp_points_per_game,"
            "pace,offensive_rating,defensive_rating,net_rating)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            TEAM_SEASON_STATS,
        )

        run_at = datetime.now(UTC).isoformat()
        con.execute(
            "INSERT INTO ingestion_log(run_at, source, records_added, notes)"
            " VALUES (?,?,?,?)",
            (run_at, "seed", _SEED_RECORD_COUNT, "Hardcoded seed data for testing"),
        )

        con.commit()
        print(f"Database built (seed mode): {db_path}")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Real ingestion via nba_api
# ---------------------------------------------------------------------------

# 2024-25 assignments; stable enough for the 2020-25 window.
_CONFERENCE: dict[str, str] = {
    "ATL": "East", "BOS": "East", "BKN": "East", "CHA": "East", "CHI": "East",
    "CLE": "East", "DET": "East", "IND": "East", "MIA": "East", "MIL": "East",
    "NYK": "East", "ORL": "East", "PHI": "East", "TOR": "East", "WAS": "East",
    "DAL": "West", "DEN": "West", "GSW": "West", "HOU": "West", "LAC": "West",
    "LAL": "West", "MEM": "West", "MIN": "West", "NOP": "West", "OKC": "West",
    "PHX": "West", "POR": "West", "SAC": "West", "SAS": "West", "UTA": "West",
}
_DIVISION: dict[str, str] = {
    "BOS": "Atlantic", "BKN": "Atlantic", "NYK": "Atlantic",
    "PHI": "Atlantic", "TOR": "Atlantic",
    "CHI": "Central", "CLE": "Central", "DET": "Central",
    "IND": "Central", "MIL": "Central",
    "ATL": "Southeast", "CHA": "Southeast", "MIA": "Southeast",
    "ORL": "Southeast", "WAS": "Southeast",
    "DEN": "Northwest", "MIN": "Northwest", "OKC": "Northwest",
    "POR": "Northwest", "UTA": "Northwest",
    "GSW": "Pacific", "LAC": "Pacific", "LAL": "Pacific",
    "PHX": "Pacific", "SAC": "Pacific",
    "DAL": "Southwest", "HOU": "Southwest", "MEM": "Southwest",
    "NOP": "Southwest", "SAS": "Southwest",
}


def _with_retries(
    fn: Any,
    *,
    attempts: int = 4,
    base_delay: float = 2.0,
    label: str = "api",
) -> Any:
    """Call *fn* with retry + exponential backoff. Raises the final exception."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if i == attempts - 1:
                break
            delay = base_delay * (2**i)
            print(f"  [{label}] attempt {i + 1} failed: {exc}; retrying in {delay:.1f}s")
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _save_raw(name: str, payload: Any, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{name}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def _row_dicts(result_set: Any) -> list[dict[str, Any]]:
    headers: list[str] = list(result_set["headers"])
    rows: list[list[Any]] = list(result_set["rowSet"])
    return [dict(zip(headers, r, strict=True)) for r in rows]


def _season_years(season_id: str) -> tuple[int, int]:
    start = int(season_id.split("-")[0])
    return start, start + 1


def build_real(
    db_path: Path = DB_PATH,
    seasons: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
) -> dict[str, int]:
    """Pull real NBA data via nba_api. Returns a dict of {table: rows_added}.

    This is an online operation. It is NOT invoked by the default test suite.
    """
    from nba_api.stats.endpoints import (  # type: ignore[import-untyped]
        leaguedashplayerstats,
        leaguedashteamstats,
    )
    from nba_api.stats.static import teams as static_teams  # type: ignore[import-untyped]

    seasons = seasons or REAL_SEASONS
    raw_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    counts: dict[str, int] = {}
    try:
        _init_schema(con)
        with con:
            # --- teams ---
            teams_raw = static_teams.get_teams()
            _save_raw("teams", teams_raw, raw_dir)
            team_rows: list[tuple[Any, ...]] = []
            for t in teams_raw:
                abbr = t["abbreviation"]
                conf = _CONFERENCE.get(abbr, "East")
                div = _DIVISION.get(abbr, "Atlantic")
                team_rows.append(
                    (t["id"], abbr, t["full_name"], t["city"], conf, div)
                )
            con.executemany(
                "INSERT OR REPLACE INTO teams VALUES (?,?,?,?,?,?)", team_rows
            )
            counts["teams"] = len(team_rows)

            # --- seasons ---
            season_rows = [(sid, *_season_years(sid)) for sid in seasons]
            con.executemany(
                "INSERT OR REPLACE INTO seasons VALUES (?,?,?)", season_rows
            )
            counts["seasons"] = len(season_rows)

            # --- per-season player + team stats ---
            player_count = 0
            pss_count = 0
            tss_count = 0
            seen_players: set[int] = set()

            for season_id in seasons:
                print(f"  fetching player stats for {season_id} ...")
                ps = _with_retries(
                    lambda sid=season_id: leaguedashplayerstats.LeagueDashPlayerStats(
                        season=sid
                    ).get_dict(),
                    label=f"player_stats_{season_id}",
                )
                _save_raw(f"player_stats_{season_id}", ps, raw_dir)
                rows = _row_dicts(ps["resultSets"][0])

                player_tuples: list[tuple[Any, ...]] = []
                pss_tuples: list[tuple[Any, ...]] = []
                for r in rows:
                    pid = int(r["PLAYER_ID"])
                    if pid not in seen_players:
                        seen_players.add(pid)
                        name = str(r["PLAYER_NAME"]).strip()
                        parts = name.split(" ", 1)
                        first = parts[0]
                        last = parts[1] if len(parts) > 1 else ""
                        player_tuples.append(
                            (pid, name, first, last, None, None, None, None, None, None)
                        )
                    pss_tuples.append(
                        (
                            pid,
                            season_id,
                            int(r["TEAM_ID"]),
                            r.get("GP"),
                            r.get("GS"),
                            r.get("MIN"),
                            r.get("PTS"),
                            r.get("REB"),
                            r.get("AST"),
                            r.get("STL"),
                            r.get("BLK"),
                            r.get("TOV"),
                            r.get("FG_PCT"),
                            r.get("FG3_PCT"),
                            r.get("FT_PCT"),
                            None,  # true_shooting_pct — advanced, not in this endpoint
                            None,  # effective_fg_pct
                            None,  # usage_rate
                            None,  # player_efficiency_rating
                            None,  # win_shares
                            None,  # box_plus_minus
                            None,  # vorp
                        )
                    )

                if player_tuples:
                    con.executemany(
                        "INSERT OR IGNORE INTO players VALUES (?,?,?,?,?,?,?,?,?,?)",
                        player_tuples,
                    )
                    player_count += len(player_tuples)

                if pss_tuples:
                    con.executemany(
                        "INSERT OR REPLACE INTO player_season_stats"
                        "(player_id,season_id,team_id,games_played,games_started,"
                        "minutes_per_game,points_per_game,rebounds_per_game,assists_per_game,"
                        "steals_per_game,blocks_per_game,turnovers_per_game,"
                        "fg_pct,fg3_pct,ft_pct,true_shooting_pct,effective_fg_pct,"
                        "usage_rate,player_efficiency_rating,win_shares,box_plus_minus,vorp)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        pss_tuples,
                    )
                    pss_count += len(pss_tuples)

                print(f"  fetching team stats for {season_id} ...")
                ts = _with_retries(
                    lambda sid=season_id: leaguedashteamstats.LeagueDashTeamStats(
                        season=sid
                    ).get_dict(),
                    label=f"team_stats_{season_id}",
                )
                _save_raw(f"team_stats_{season_id}", ts, raw_dir)
                trows = _row_dicts(ts["resultSets"][0])

                tss_tuples = [
                    (
                        int(r["TEAM_ID"]),
                        season_id,
                        r.get("W"),
                        r.get("L"),
                        r.get("PTS"),
                        None,  # opp_points_per_game — needs opp endpoint
                        None,  # pace
                        None,  # offensive_rating
                        None,  # defensive_rating
                        None,  # net_rating
                    )
                    for r in trows
                ]
                if tss_tuples:
                    con.executemany(
                        "INSERT OR REPLACE INTO team_season_stats"
                        "(team_id,season_id,wins,losses,points_per_game,"
                        "opp_points_per_game,pace,offensive_rating,"
                        "defensive_rating,net_rating)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?)",
                        tss_tuples,
                    )
                    tss_count += len(tss_tuples)

            counts["players"] = player_count
            counts["player_season_stats"] = pss_count
            counts["team_season_stats"] = tss_count

            total = sum(counts.values())
            con.execute(
                "INSERT INTO ingestion_log(run_at, source, records_added, notes)"
                " VALUES (?,?,?,?)",
                (
                    datetime.now(UTC).isoformat(),
                    "nba_api",
                    total,
                    f"seasons={','.join(seasons)}; counts={counts}",
                ),
            )
        print(f"Database built (real mode): {db_path}")
        print(f"Counts: {counts}")
    finally:
        con.close()
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the NBA stats SQLite database.")
    p.add_argument(
        "--mode",
        choices=["seed", "real"],
        default="seed",
        help="seed = fast offline fixture (default). real = pull via nba_api.",
    )
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--raw", type=Path, default=RAW_DIR)
    p.add_argument(
        "--seasons",
        nargs="+",
        default=REAL_SEASONS,
        help="Seasons for real mode. Default: 2020-21..2024-25.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.mode == "seed":
        build(db_path=args.db)
    else:
        build_real(db_path=args.db, seasons=args.seasons, raw_dir=args.raw)


if __name__ == "__main__":
    main()
