from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Top 5 European leagues (football-data codes) x recent seasons (folder codes)
DEFAULT_SEASONS: tuple[str, ...] = ("2122", "2223", "2324", "2425")
DEFAULT_DIVS: tuple[str, ...] = ("E0", "SP1", "I1", "D1", "F1")


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    rows: int


def season_csv_url(season: str, div: str) -> str:
    return f"{BASE_URL}/{season}/{div}.csv"


def download_csv(url: str, dest: Path, timeout_s: int = 60) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted public football-data URL)


def download_default_raw_data(raw_dir: Path) -> list[DownloadResult]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    for season in DEFAULT_SEASONS:
        for div in DEFAULT_DIVS:
            url = season_csv_url(season, div)
            dest = raw_dir / f"{season}_{div}.csv"
            download_csv(url, dest)
            n = sum(1 for _ in dest.open("rb")) - 1
            results.append(DownloadResult(path=dest, rows=n))
    return results


def load_raw_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p, encoding="latin-1")
        df["__source_file"] = p.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def _parse_date(df: pd.DataFrame) -> pd.Series:
    if "Time" in df.columns:
        dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), dayfirst=True, errors="coerce")
    else:
        dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return dt


def _match_points_home(row: pd.Series) -> int:
    if row["FTHG"] > row["FTAG"]:
        return 3
    if row["FTHG"] == row["FTAG"]:
        return 1
    return 0


def _match_points_away(row: pd.Series) -> int:
    if row["FTAG"] > row["FTHG"]:
        return 3
    if row["FTAG"] == row["FTHG"]:
        return 1
    return 0


def build_long_team_games(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    m["DateTime"] = _parse_date(m)
    m = m.dropna(subset=["DateTime", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    home = pd.DataFrame(
        {
            "Div": m["Div"],
            "DateTime": m["DateTime"],
            "team": m["HomeTeam"],
            "opponent": m["AwayTeam"],
            "is_home": 1,
            "goals_for": m["FTHG"],
            "goals_against": m["FTAG"],
            "points": m.apply(_match_points_home, axis=1),
        }
    )
    away = pd.DataFrame(
        {
            "Div": m["Div"],
            "DateTime": m["DateTime"],
            "team": m["AwayTeam"],
            "opponent": m["HomeTeam"],
            "is_home": 0,
            "goals_for": m["FTAG"],
            "goals_against": m["FTHG"],
            "points": m.apply(_match_points_away, axis=1),
        }
    )
    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team", "DateTime"], kind="mergesort")
    return long_df


def add_shifted_rolling_team_features(long_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = long_df.copy()
    g = out.groupby("team", sort=False)

    out["gf_roll"] = g["goals_for"].transform(lambda s: s.rolling(window, min_periods=1).mean().shift(1))
    out["ga_roll"] = g["goals_against"].transform(lambda s: s.rolling(window, min_periods=1).mean().shift(1))
    out["pts_roll"] = g["points"].transform(lambda s: s.rolling(window, min_periods=1).mean().shift(1))
    out["home_share_roll"] = g["is_home"].transform(lambda s: s.rolling(window, min_periods=1).mean().shift(1))

    return out


def attach_team_features_to_matches(matches: pd.DataFrame, long_with_features: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    m["DateTime"] = _parse_date(m)
    m = m.dropna(subset=["DateTime", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    home_key = long_with_features.rename(
        columns={
            "team": "HomeTeam",
            "gf_roll": "home_gf_roll",
            "ga_roll": "home_ga_roll",
            "pts_roll": "home_pts_roll",
            "home_share_roll": "home_home_share_roll",
        }
    )[["Div", "DateTime", "HomeTeam", "home_gf_roll", "home_ga_roll", "home_pts_roll", "home_home_share_roll"]]

    away_key = long_with_features.rename(
        columns={
            "team": "AwayTeam",
            "gf_roll": "away_gf_roll",
            "ga_roll": "away_ga_roll",
            "pts_roll": "away_pts_roll",
            "home_share_roll": "away_home_share_roll",
        }
    )[["Div", "DateTime", "AwayTeam", "away_gf_roll", "away_ga_roll", "away_pts_roll", "away_home_share_roll"]]

    out = m.merge(home_key, on=["Div", "DateTime", "HomeTeam"], how="left")
    out = out.merge(away_key, on=["Div", "DateTime", "AwayTeam"], how="left")
    return out


def implied_probs_from_odds(odds_h: pd.Series, odds_d: pd.Series, odds_a: pd.Series) -> pd.DataFrame:
    inv_h = 1.0 / odds_h
    inv_d = 1.0 / odds_d
    inv_a = 1.0 / odds_a
    overround = inv_h + inv_d + inv_a
    return pd.DataFrame(
        {
            "imp_h": inv_h / overround,
            "imp_d": inv_d / overround,
            "imp_a": inv_a / overround,
            "overround": overround,
        }
    )


def build_model_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Pre-match feature table: rolling history + closing odds (no same-match goals/stats)."""
    required = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    odds_cols = ("B365H", "B365D", "B365A")
    if not all(c in matches.columns for c in odds_cols):
        raise ValueError("Expected Bet365 closing odds columns B365H/B365D/B365A (standard in football-data).")

    long = build_long_team_games(matches)
    long = add_shifted_rolling_team_features(long, window=5)
    df = attach_team_features_to_matches(matches, long)

    odds = df[list(odds_cols)].apply(pd.to_numeric, errors="coerce")
    probs = implied_probs_from_odds(odds["B365H"], odds["B365D"], odds["B365A"])
    df = pd.concat([df, probs], axis=1)

    df["y_ftr"] = df["FTR"].astype(str).str.upper().str.strip()
    df = df[df["y_ftr"].isin(["H", "D", "A"])].copy()

    # Drop rows without usable odds or without any prior team history
    df = df.dropna(
        subset=[
            "imp_h",
            "imp_d",
            "imp_a",
            "home_gf_roll",
            "home_ga_roll",
            "home_pts_roll",
            "away_gf_roll",
            "away_ga_roll",
            "away_pts_roll",
        ]
    )

    df["pts_diff_roll"] = df["home_pts_roll"] - df["away_pts_roll"]
    df["gf_diff_roll"] = df["home_gf_roll"] - df["away_gf_roll"]
    df["ga_diff_roll"] = df["home_ga_roll"] - df["away_ga_roll"]

    return df.reset_index(drop=True)


def ftr_to_int(y: pd.Series) -> np.ndarray:
    mapping = {"H": 0, "D": 1, "A": 2}
    return y.map(mapping).to_numpy(dtype=np.int64)


def int_to_ftr(y: np.ndarray | list[int]) -> list[str]:
    inv = {0: "H", 1: "D", 2: "A"}
    return [inv[int(i)] for i in y]
