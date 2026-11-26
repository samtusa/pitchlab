"""PitchLab web API + static frontend."""
from __future__ import annotations

import json
from typing import Optional
import os
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pitchlab.config import MIN_PITCHES, PROCESSED_DIR

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="PitchLab")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_cache: dict[tuple, tuple[float, object]] = {}


def _load(kind: str, season: int):
    ext = "json" if kind == "meta" else "parquet"
    path = PROCESSED_DIR / f"{kind}_{season}.{ext}"
    if not path.exists():
        raise HTTPException(404, f"No {kind} data for {season}")
    mtime = path.stat().st_mtime
    hit = _cache.get((kind, season))
    if hit and hit[0] == mtime:
        return hit[1]
    obj = json.loads(path.read_text()) if ext == "json" else pd.read_parquet(path)
    _cache[(kind, season)] = (mtime, obj)
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.replace({np.nan: None})
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


def seasons() -> list[int]:
    return sorted((int(p.stem.split("_")[1]) for p in PROCESSED_DIR.glob("meta_*.json")), reverse=True)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/seasons")
def api_seasons():
    return seasons()


@app.get("/api/meta/{season}")
def api_meta(season: int):
    return _load("meta", season)


SORTS = {"slot_outlier", "ivb_oe", "hb_oe", "perceived_velo", "velo_gain", "velo", "extension", "arm_angle",
         "stuff_plus", "rv100", "whiff_pct", "vaa"}


@app.get("/api/leaderboard")
def leaderboard(season: int, family: Optional[str] = None, hand: Optional[str] = None,
                min_pitches: int = MIN_PITCHES, sort: str = "slot_outlier",
                ascending: bool = False, limit: int = Query(50, le=3000)):
    if sort not in SORTS:
        raise HTTPException(400, f"sort must be one of {sorted(SORTS)}")
    a = _load("arsenal", season)
    q = a[a["n"] >= min_pitches]
    if family:
        q = q[q["family"] == family]
    if hand in ("L", "R"):
        q = q[q["p_throws"] == hand]
    q = q.dropna(subset=[sort]).sort_values(sort, ascending=ascending).head(limit)
    return _records(q)


@app.get("/api/scatter")
def scatter(season: int, family: str, min_pitches: int = MIN_PITCHES):
    a = _load("arsenal", season)
    q = a[(a["family"] == family) & (a["n"] >= min_pitches)]
    cols = ["pitcher", "player_name", "p_throws", "pitch_type", "n", "arm_angle", "ivb", "hb",
            "exp_ivb", "exp_hb", "ivb_oe", "hb_oe", "slot_outlier", "velo", "stuff_plus", "rv100", "whiff_pct"]
    cols = [c for c in cols if c in a.columns]
    curve = _load("meta", season).get("models", {}).get(family)
    return {"points": _records(q[cols]), "curve": curve}


@app.get("/api/analysis")
def analysis():
    path = PROCESSED_DIR / "analysis.json"
    if not path.exists():
        raise HTTPException(404, "No analysis built yet")
    return json.loads(path.read_text())


@app.get("/api/pitcher/{pid}")
def pitcher(pid: int, season: int):
    a = _load("arsenal", season)
    p = _load("pitchers", season)
    s = p[p["pitcher"] == pid]
    if s.empty:
        raise HTTPException(404, "Pitcher not found")
    ars = a[a["pitcher"] == pid].sort_values("n", ascending=False)
    return {"summary": _records(s)[0], "arsenal": _records(ars)}


@app.get("/api/search")
def search(season: int, q: str = Query(..., min_length=2)):
    p = _load("pitchers", season)
    hits = p[p["player_name"].str.contains(q, case=False, na=False)]
    return _records(hits.sort_values("pitches", ascending=False).head(15)[["pitcher", "player_name", "p_throws", "pitches"]])


# ---- data refresh ---------------------------------------------------------

_refresh_lock = threading.Lock()


def _refresh(season: int):
    if not _refresh_lock.acquire(blocking=False):
        return
    try:
        from pitchlab.build import build
        from pitchlab.config import RAW_DIR
        from pitchlab.ingest import load_raw, update_season
        # Refresh this season, then rebuild every cached season together so
        # Stuff+ is trained on all of them.
        update_season(season)
        seasons = sorted(int(p.stem.split("_")[1]) for p in RAW_DIR.glob("statcast_*.parquet"))
        raws = {s: load_raw(s) for s in seasons}
        raws = {s: r for s, r in raws.items() if not r.empty}
        if raws:
            build(raws, "statcast")
    finally:
        _refresh_lock.release()


@app.post("/api/refresh")
def refresh(season: int, background: BackgroundTasks, x_refresh_token: Optional[str] = Header(None)):
    token = os.environ.get("PITCHLAB_REFRESH_TOKEN")
    if not token or x_refresh_token != token:
        raise HTTPException(403, "Bad or missing refresh token")
    background.add_task(_refresh, season)
    return {"status": "started", "season": season}


def _daily_loop():
    """During the season, pull new games once a day (around 10:00 UTC)."""
    while True:
        time.sleep(3600)
        today = date.today()
        if 3 <= today.month <= 10 and time.gmtime().tm_hour == 10:
            try:
                _refresh(today.year)
            except Exception as e:  # keep the loop alive
                print("daily refresh failed:", e)


if os.environ.get("PITCHLAB_AUTO_REFRESH") == "1":
    threading.Thread(target=_daily_loop, daemon=True).start()
