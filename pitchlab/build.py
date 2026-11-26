"""Build processed tables.

Usage:
    python -m pitchlab.build 2025 2026            # download from Savant, then build
    python -m pitchlab.build 2025 2026 --no-fetch # rebuild from cached raw data
    python -m pitchlab.build 2025 2026 --sample   # synthetic data, no network

Stuff+ is trained on every season given together, so pass all seasons you
want graded in one command.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from .config import PROCESSED_DIR
from .features import add_features
from .metrics import add_pitch_moe, aggregate_arsenal, fit_movement_models, pitcher_summary


def build(raws: dict, source: str, log=print) -> dict:
    """raws: {season: raw DataFrame}. Writes processed files and returns metas."""
    from . import stuff

    pitches_all, arsenals, curves_all, ext = [], {}, {}, {}
    for season, raw in sorted(raws.items()):
        p = add_features(raw)
        ext[season] = p.attrs.get("league_extension_ft", float("nan"))
        a, curves = fit_movement_models(aggregate_arsenal(p))
        p = add_pitch_moe(p, curves)
        pitches_all.append(p)
        arsenals[season] = a
        curves_all[season] = curves.get(season, {})
    pitches = pd.concat(pitches_all, ignore_index=True)

    report = {}
    if stuff.has_outcomes(pitches):
        log("training Stuff+ models (a few minutes)")
        graded, report = stuff.fit_stuff(pitches, log=log)
        sp = stuff.stuff_plus_table(graded)
        for season in arsenals:
            arsenals[season] = arsenals[season].merge(sp, on=["game_year", "pitcher", "pitch_type"], how="left")
    else:
        log("no outcome columns in raw data; skipping Stuff+")

    metas = {}
    for season, a in arsenals.items():
        pitchers = pitcher_summary(a)
        a.to_parquet(PROCESSED_DIR / f"arsenal_{season}.parquet", index=False)
        pitchers.to_parquet(PROCESSED_DIR / f"pitchers_{season}.parquet", index=False)
        sp_ = pitches[pitches["game_year"] == season]
        meta = {
            "season": int(season),
            "source": source,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pitches": int(len(sp_)),
            "pitchers": int(pitchers["pitcher"].nunique()),
            "league_extension_ft": round(ext[season], 2),
            "arm_angle_estimated_share": round(float(sp_["arm_angle_estimated"].mean()), 3),
            "has_stuff": bool(report),
            "models": curves_all[season],
        }
        (PROCESSED_DIR / f"meta_{season}.json").write_text(json.dumps(meta, indent=2))
        metas[season] = meta
        log(f"built {season}: {meta['pitches']:,} pitches, {meta['pitchers']} pitchers ({source})")

    all_arsenal = pd.concat(arsenals.values(), ignore_index=True)
    analysis = {
        "seasons": sorted(int(s) for s in arsenals),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "validation": report,
        "quintiles": stuff.quintile_analysis(all_arsenal),
    }
    (PROCESSED_DIR / "analysis.json").write_text(json.dumps(analysis, indent=2))
    return metas


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("seasons", type=int, nargs="+")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--sample", action="store_true")
    a = p.parse_args(argv)

    if a.sample:
        from .sample import make_sample
        from .ingest import raw_path
        raws = {s: make_sample(s) for s in a.seasons}
        for s, r in raws.items():
            r.to_parquet(raw_path(s), index=False)
        build(raws, "sample")
        return
    from .ingest import load_raw, update_season
    raws = {}
    for s in a.seasons:
        raw = load_raw(s) if a.no_fetch else update_season(s)
        if raw.empty:
            raise SystemExit(f"No raw data for {s}")
        raws[s] = raw
    build(raws, "statcast")


if __name__ == "__main__":
    main()
