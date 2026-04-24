"""Build the NBA stats SQLite database with seed data."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "nba.db"

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
"""

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

# (season_id, start_year, end_year)
SEASONS = [
    ("2022-23", 2022, 2023),
    ("2023-24", 2023, 2024),
]

# (game_id, game_date, season_id, home_team_id, away_team_id,
#  home_score, away_score, game_type)
GAMES = [
    (1, "2023-01-15", "2022-23", 1, 2, 118, 110, "regular"),
    (2, "2023-02-20", "2022-23", 3, 1, 112, 120, "regular"),
    (3, "2024-01-10", "2023-24", 2, 3, 125, 119, "regular"),
    (4, "2024-03-05", "2023-24", 1, 3, 107,  98, "regular"),
]

# (game_id, player_id, team_id, minutes, points, rebounds, assists,
#  steals, blocks, turnovers, fg_made, fg_attempted, fg3_made, fg3_attempted,
#  ft_made, ft_attempted, plus_minus)
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

# (player_id, season_id, team_id, games_played, games_started,
#  minutes_per_game, points_per_game, rebounds_per_game, assists_per_game,
#  steals_per_game, blocks_per_game, turnovers_per_game,
#  fg_pct, fg3_pct, ft_pct,
#  true_shooting_pct, effective_fg_pct, usage_rate,
#  player_efficiency_rating, win_shares, box_plus_minus, vorp)
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

# (team_id, season_id, wins, losses, points_per_game, opp_points_per_game,
#  pace, offensive_rating, defensive_rating, net_rating)
TEAM_SEASON_STATS = [
    (1, "2022-23", 57, 25, 117.9, 112.5, 98.3, 116.3, 110.9,  5.4),
    (2, "2022-23", 44, 38, 120.2, 115.6, 99.8, 118.3, 113.9,  4.4),
    (3, "2022-23", 43, 39, 117.2, 116.4, 99.0, 115.5, 114.8,  0.7),
    (1, "2023-24", 64, 18, 120.6, 110.5, 99.1, 122.2, 112.0, 10.2),
    (2, "2023-24", 46, 36, 118.4, 115.7, 99.4, 117.6, 115.0,  2.6),
    (3, "2023-24", 47, 35, 118.3, 116.2, 99.7, 116.8, 114.7,  2.1),
]


def build(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)

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
        con.commit()
        print(f"Database built: {db_path}")
    finally:
        con.close()


if __name__ == "__main__":
    build()
