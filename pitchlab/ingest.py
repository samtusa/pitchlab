"""Download pitch-by-pitch Statcast data from Baseball Savant.

Savant caps each CSV response at about 25,000 rows, so we request a few days
at a time. Results are cached per season in data/raw/statcast_<year>.parquet,
and a re-run only fetches dates newer than what is already stored.
"""
from __future__ import annotations

import io
import time
from datetime import date, timedelta

import pandas as pd
import requests

from .config import RAW_DIR, REQUIRED_RAW_COLUMN, STATCAST_COLUMNS

SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
CHUNK_DAYS = 3
HEADERS = {"User-Agent": "Mozilla/5.0 (PitchLab research tool)"}


def _fetch_range(start: date, end: date, retries: int = 3) -> pd.DataFrame:
    params = {
        "all": "true",
        "type": "details",
        "player_type": "pitcher",
        "hfGT": "R|",  # regular season only
        "game_date_gt": start.isoformat(),
        "game_date_lt": end.isoformat(),
    }
    for attempt in range(retries):
        try:
            r = requests.get(SAVANT_URL, params=params, headers=HEADERS, timeout=120)
            r.raise_for_status()
            text = r.content.decode("utf-8-sig")
            if not text.strip():
                return pd.DataFrame()
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            if len(df) >= 25000:
                print(f"  warning: {start}..{end} hit the 25k row cap; data may be truncated")
            keep = [c for c in STATCAST_COLUMNS if c in df.columns]
            return df[keep]
        except (requests.RequestException, pd.errors.ParserError) as e:
            wait = 5 * (attempt + 1)
            print(f"  {start}..{end} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch {start}..{end} from Savant")


def raw_path(season: int):
    return RAW_DIR / f"statcast_{season}.parquet"


def load_raw(season: int) -> pd.DataFrame:
    p = raw_path(season)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def update_season(season: int, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Fetch any missing dates for a season and return the full season table."""
    start = start or date(season, 3, 15)
    end = min(end or date(season, 11, 5), date.today() - timedelta(days=1))

    existing = load_raw(season)
    if not existing.empty and REQUIRED_RAW_COLUMN not in existing.columns:
        print(f"cached {season} data is missing outcome columns; downloading the season again")
        existing = pd.DataFrame()
    if not existing.empty:
        last = pd.to_datetime(existing["game_date"]).max().date()
        start = max(start, last + timedelta(days=1))

    frames = [existing] if not existing.empty else []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        print(f"fetching {cur} .. {chunk_end}")
        df = _fetch_range(cur, chunk_end)
        if not df.empty:
            frames.append(df)
        cur = chunk_end + timedelta(days=1)
        time.sleep(1.5)  # be polite to Savant

    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(raw_path(season), index=False)
    print(f"season {season}: {len(full):,} pitches stored")
    return full
