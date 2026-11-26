"""Robustness checks and reader-friendly numbers for pitch-movement write-ups.

Run after a build:
    python -m pitchlab.writeup

Writes data/writeup/:
    results.md    readable summary to paste back into the chat
    results.json  every number, for charts and tables later
    *.png         quintile charts with 95% error bars for the headline findings

Outcomes (computed from the raw pitch data, so no rebuild is needed)
    whiff_pct    swings and misses per swing           (can they make contact?)
    chase_pct    swings at pitches outside the zone    (are they fooled?)
    gb_pct       ground balls per ball in play          (what kind of contact?)
    popup_pct    pop-ups per ball in play
    hardhit_pct  balls in play hit 95+ mph              (how damaging?)
Appendix: xwOBA on contact and run value per 100 pitches.

Traits are oriented so "more" reads naturally:
    ride_oe     IVB over slot expectation (in)
    run_oe      arm-side break over slot expectation (in)   [fastballs, changeups, splitters]
    sweep_oe    glove-side break over slot expectation (in) [sliders, sweepers, cutters, curveballs]
    flat_vaa    approach angle vs expected at that height (deg, + = flatter)
    sep_velo    mph slower than his primary fastball
    sep_drop    inches more drop than his primary fastball
    sep_hb      inches more arm-side break than his primary fastball

For every pitch type x trait x outcome:
  - top 20% vs bottom 20% of pitch types, with a 95% interval from a bootstrap
    that resamples PITCHERS (so a pitcher's two seasons count as one unit)
  - effect per unit of the trait with controls applied (velocity; for fastball
    ride also approach angle and vice versa; for separation, fastball velocity)
  - the per-unit effect in each season separately
  - a verdict: SOLID / SUGGESTIVE / NOISE
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR
from .features import SWINGS

OUT = PROCESSED_DIR.parent / "writeup"
QUALIFY = 100
EXAMPLE_MIN = 300
REPS = 500

FASTBALLS = ("Four-Seam", "Sinker")
GLOVE_SIDE = ("Slider", "Sweeper", "Cutter", "Curveball")

TRAITS = {
    "ride_oe": ("Ride over slot expectation", "in"),
    "run_oe": ("Arm-side run over slot expectation", "in"),
    "sweep_oe": ("Glove-side sweep over slot expectation", "in"),
    "flat_vaa": ("Flatness vs. expected approach angle", "deg"),
    "sep_velo": ("Speed gap below his fastball", "mph"),
    "sep_drop": ("Extra drop vs. his fastball", "in"),
    "sep_hb": ("Extra arm-side break vs. his fastball", "in"),
}
OUTCOMES = {
    # key: (label, display multiplier, unit, weight column, higher_is_better or None if neutral)
    "whiff_pct": ("Whiff rate", 100, "pts", "swings", True),
    "chase_pct": ("Chase rate", 100, "pts", "out_zone", True),
    "gb_pct": ("Ground-ball rate", 100, "pts", "bbe", None),
    "popup_pct": ("Pop-up rate", 100, "pts", "bbe", True),
    "hardhit_pct": ("Hard-hit rate (95+ mph)", 100, "pts", "bbe_ev", False),
    "xwoba_con": ("xwOBA on contact", 1000, "pts of xwOBA", "contacts", False),
    "rv100": ("Run value / 100 pitches", 1, "runs per 100", "n", True),
}
MAIN_OUTCOMES = ("whiff_pct", "chase_pct", "gb_pct", "popup_pct", "hardhit_pct")
APPENDIX_OUTCOMES = ("xwoba_con", "rv100")

# Headline charts: (family, trait)
HEADLINES = [("Four-Seam", "ride_oe"), ("Four-Seam", "flat_vaa"), ("Sweeper", "sweep_oe"),
             ("Slider", "ride_oe"), ("Changeup", "sep_velo"), ("Slider", "sep_velo")]


def contact_table() -> pd.DataFrame:
    """Chase, batted-ball type and hard-hit rates per season x pitcher x pitch type."""
    need = ["game_year", "pitcher", "pitch_type", "description", "zone", "bb_type", "launch_speed"]
    frames = []
    for f in sorted(RAW_DIR.glob("statcast_*.parquet")):
        import pyarrow.parquet as pq
        cols = [c for c in need if c in pq.read_schema(f).names]
        frames.append(pd.read_parquet(f, columns=cols))
    if not frames:
        return pd.DataFrame()
    r = pd.concat(frames, ignore_index=True)
    for c in need:
        if c not in r:
            r[c] = np.nan
    swing = r["description"].isin(SWINGS)
    out_zone = r["zone"].between(11, 14)
    bbe = r["bb_type"].notna()
    ev = bbe & r["launch_speed"].notna()
    r = r.assign(_oz=out_zone, _chase=swing & out_zone, _bbe=bbe,
                 _gb=r["bb_type"].eq("ground_ball"), _pu=r["bb_type"].eq("popup"),
                 _ev=ev, _hh=ev & (r["launch_speed"] >= 95))
    g = r.groupby(["game_year", "pitcher", "pitch_type"]).agg(
        out_zone=("_oz", "sum"), chases=("_chase", "sum"), bbe=("_bbe", "sum"),
        gbs=("_gb", "sum"), pus=("_pu", "sum"), bbe_ev=("_ev", "sum"), hhs=("_hh", "sum")).reset_index()
    g["chase_pct"] = g["chases"] / g["out_zone"].where(g["out_zone"] > 0)
    g["gb_pct"] = g["gbs"] / g["bbe"].where(g["bbe"] > 0)
    g["popup_pct"] = g["pus"] / g["bbe"].where(g["bbe"] > 0)
    g["hardhit_pct"] = g["hhs"] / g["bbe_ev"].where(g["bbe_ev"] > 0)
    return g[["game_year", "pitcher", "pitch_type", "out_zone", "bbe", "bbe_ev",
              "chase_pct", "gb_pct", "popup_pct", "hardhit_pct"]]


def prepare(a: pd.DataFrame, contact: pd.DataFrame | None = None) -> pd.DataFrame:
    a = a.copy()
    if contact is not None and len(contact):
        a = a.merge(contact, on=["game_year", "pitcher", "pitch_type"], how="left")
    for c in ("out_zone", "bbe", "bbe_ev", "chase_pct", "gb_pct", "popup_pct", "hardhit_pct"):
        if c not in a:
            a[c] = np.nan
    glove = a["family"].isin(GLOVE_SIDE)
    a["ride_oe"] = a["ivb_oe"]
    a["run_oe"] = a["hb_oe"].where(~glove)
    a["sweep_oe"] = (-a["hb_oe"]).where(glove)
    a["flat_vaa"] = a["vaa_adj"] if "vaa_adj" in a else np.nan
    a["sep_velo"] = -a["d_velo"]
    a["sep_drop"] = -a["d_ivb"]
    a["sep_hb"] = a["d_hb"]
    a["fb_velo"] = a["velo"] - a["d_velo"]
    a["contacts"] = (a["swings"] - a["whiffs"]).clip(lower=0)
    # separation is undefined for a pitcher's primary fastball itself
    is_fb = a["d_velo"].eq(0)
    a.loc[is_fb, ["sep_velo", "sep_drop", "sep_hb"]] = np.nan
    return a


def traits_for(fam: str) -> list[str]:
    side = "sweep_oe" if fam in GLOVE_SIDE else "run_oe"
    extra = ["flat_vaa"] if fam in FASTBALLS else ["sep_velo", "sep_drop", "sep_hb"]
    return ["ride_oe", side] + extra


def controls_for(fam: str, trait: str) -> list[str]:
    if trait.startswith("sep_"):
        return ["fb_velo"] if trait == "sep_velo" else ["velo", "fb_velo"]
    c = ["velo"]
    if fam in FASTBALLS and trait == "ride_oe":
        c.append("flat_vaa")
    if trait == "flat_vaa":
        c.append("ride_oe")
    return c


def _clean(d, cols, wcol):
    m = np.ones(len(d), bool)
    for c in cols + [wcol]:
        m &= np.isfinite(d[c].to_numpy(float))
    m &= d[wcol].to_numpy(float) > 0
    return m


def slope(d, x, y, controls, wcol):
    """Weighted regression coefficient of y on x, holding controls fixed."""
    m = _clean(d, [x, y] + controls, wcol)
    if m.sum() < 15:
        return np.nan
    X = np.column_stack([np.ones(m.sum()), d[x].to_numpy(float)[m]] +
                        [d[c].to_numpy(float)[m] for c in controls])
    w = np.sqrt(d[wcol].to_numpy(float)[m])
    beta, *_ = np.linalg.lstsq(X * w[:, None], d[y].to_numpy(float)[m] * w, rcond=None)
    return float(beta[1])


def wcorr(d, x, y, wcol) -> float:
    m = _clean(d, [x, y], wcol)
    if m.sum() < 10:
        return np.nan
    xv, yv, w = d[x].to_numpy(float)[m], d[y].to_numpy(float)[m], d[wcol].to_numpy(float)[m]
    mx, my = np.average(xv, weights=w), np.average(yv, weights=w)
    vx, vy = np.average((xv - mx) ** 2, weights=w), np.average((yv - my) ** 2, weights=w)
    return float(np.average((xv - mx) * (yv - my), weights=w) / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def quint_means(d, x, y, wcol, edges):
    q = np.digitize(d[x].to_numpy(float), edges)
    m = _clean(d, [x, y], wcol)
    out = []
    for k in range(5):
        mk = m & (q == k)
        w = d[wcol].to_numpy(float)[mk]
        out.append(float(np.average(d[y].to_numpy(float)[mk], weights=w)) if w.sum() > 0 else np.nan)
    return out


def verdict(gap_ci, slope_ci, seasons, gap):
    sig_gap = gap_ci[0] > 0 or gap_ci[1] < 0
    sig_slope = slope_ci[0] > 0 or slope_ci[1] < 0
    s = [v for v in seasons if np.isfinite(v)]
    same = len(s) >= 2 and all(np.sign(v) == np.sign(gap) for v in s)
    if sig_gap and sig_slope and same:
        return "SOLID"
    if sig_gap or sig_slope:
        return "SUGGESTIVE"
    return "NOISE"


def analyze_family(sub: pd.DataFrame, fam: str, rng):
    seasons = sorted(sub["game_year"].unique())
    pitchers = sub["pitcher"].unique()
    pid = sub["pitcher"].to_numpy()
    rows_of = {p: np.flatnonzero(pid == p) for p in pitchers}
    combos = [(t, o) for t in traits_for(fam) if sub[t].notna().sum() >= 30
              for o in OUTCOMES if sub[o].notna().sum() >= 30 and sub[OUTCOMES[o][3]].fillna(0).sum() > 0]
    edges = {t: np.nanquantile(sub[t], [0.2, 0.4, 0.6, 0.8]) for t, _ in combos}
    boot = {c: {"q": [], "slope": []} for c in combos}

    for _ in range(REPS):
        pick = rng.choice(pitchers, len(pitchers), replace=True)
        d = sub.iloc[np.concatenate([rows_of[p] for p in pick])]
        for t, o in combos:
            wcol = OUTCOMES[o][3]
            boot[(t, o)]["q"].append(quint_means(d, t, o, wcol, edges[t]))
            boot[(t, o)]["slope"].append(slope(d, t, o, controls_for(fam, t), wcol))

    results = []
    for t, o in combos:
        label, mult, unit, wcol, better_high = OUTCOMES[o]
        ctrl = controls_for(fam, t)
        q = np.array(quint_means(sub, t, o, wcol, edges[t])) * mult
        qb = np.array(boot[(t, o)]["q"], dtype=float) * mult
        gaps = qb[:, 4] - qb[:, 0]
        sl = slope(sub, t, o, ctrl, wcol) * mult
        slb = np.array(boot[(t, o)]["slope"], dtype=float) * mult
        per_season = {int(s): round(slope(sub[sub["game_year"] == s], t, o, ctrl, wcol) * mult, 3) for s in seasons}
        gap = q[4] - q[0]
        gap_ci = np.nanpercentile(gaps, [2.5, 97.5])
        sl_ci = np.nanpercentile(slb, [2.5, 97.5])
        if better_high is None:
            direction = "more" if gap > 0 else "fewer"
        else:
            direction = "better" if (gap > 0) == better_high else "worse"
        results.append({
            "family": fam, "trait": t, "outcome": o, "pitch_types": int(sub[t].notna().sum()),
            "edges": [round(float(e), 2) for e in edges[t]],
            "bottom20": round(q[0], 2), "top20": round(q[4], 2),
            "gap": round(gap, 2), "gap_ci": [round(gap_ci[0], 2), round(gap_ci[1], 2)],
            "per_unit": round(sl, 3), "per_unit_ci": [round(sl_ci[0], 3), round(sl_ci[1], 3)],
            "per_unit_by_season": per_season, "controls": ctrl,
            "r": round(wcorr(sub, t, o, wcol), 3),
            "direction": direction,
            "verdict": verdict(gap_ci, sl_ci, list(per_season.values()), gap),
            "quintiles": {"mean": [round(v, 3) for v in q],
                          "lo": [round(float(v), 3) for v in np.nanpercentile(qb, 2.5, axis=0)],
                          "hi": [round(float(v), 3) for v in np.nanpercentile(qb, 97.5, axis=0)]},
        })
    return results


def _name(n: str) -> str:
    last, _, first = str(n).partition(", ")
    return f"{first} {last}".strip()


def examples(sub: pd.DataFrame, trait: str, k: int = 5) -> dict:
    """Most extreme pitchers on a trait, one row per pitcher (his most extreme season)."""
    d = sub[(sub["n"] >= EXAMPLE_MIN) & sub[trait].notna()].copy()
    d["player_name"] = d["player_name"].map(_name)
    cols = [c for c in ["player_name", "game_year", "pitch_type", "n", "velo", trait, "whiff_pct", "chase_pct",
                        "gb_pct", "popup_pct", "hardhit_pct", "xwoba_con", "rv100", "stuff_plus"] if c in d]

    def pick(ascending):
        x = d.sort_values(trait, ascending=ascending).drop_duplicates("pitcher")
        return x.head(k)[cols].round(3).to_dict("records")
    return {"top": pick(False), "bottom": pick(True)}


def model_table(analysis: dict) -> list:
    out = []
    for fam, variants in analysis.get("validation", {}).items():
        for name, e in variants.items():
            row = {"family": fam, "model": name}
            for key in ("next_rv", "next_whiff", "next_rv_from_results", "next_whiff_from_results"):
                r, n = e.get(key), e.get("pairs")
                if r is None or not n or n < 5:
                    continue
                z, se = np.arctanh(np.clip(r, -0.999, 0.999)), 1 / np.sqrt(n - 3)
                row[key] = [round(r, 3), round(float(np.tanh(z - 1.96 * se)), 3), round(float(np.tanh(z + 1.96 * se)), 3)]
            out.append(row)
    return out


def chart(fam, trait, res_by_key, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outs = [o for o in MAIN_OUTCOMES if (trait, o) in res_by_key]
    cols = 3
    rows_n = int(np.ceil(len(outs) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 3.4 * rows_n), squeeze=False)
    name, unit = TRAITS[trait]
    for ax in axes.flat[len(outs):]:
        ax.axis("off")
    for ax, o in zip(axes.flat, outs):
        r = res_by_key[(trait, o)]
        e = r["edges"]
        labels = [f"Bottom\n20%", f"{e[0]:.1f} to\n{e[1]:.1f}", f"{e[1]:.1f} to\n{e[2]:.1f}",
                  f"{e[2]:.1f} to\n{e[3]:.1f}", f"Top\n20%"]
        m = np.array(r["quintiles"]["mean"])
        lo, hi = np.array(r["quintiles"]["lo"]), np.array(r["quintiles"]["hi"])
        # dots, not bars: the axis is zoomed, and bars would exaggerate differences
        ax.errorbar(range(5), m, yerr=[m - lo, hi - m], fmt="o", color="#2a78d6", ecolor="#2a78d6",
                    elinewidth=1.5, capsize=4, markersize=7, markeredgecolor="white", markeredgewidth=1.2)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, 4.5)
        ax.set_xticks(range(5), labels, fontsize=7.5)
        ax.set_title(f"{OUTCOMES[o][0]} ({r['verdict'].lower()})", fontsize=9.5)
        ax.set_ylabel("%", fontsize=9)
        span = np.nanmax(hi) - np.nanmin(lo)
        ax.set_ylim(max(0, np.nanmin(lo) - span * 0.6), np.nanmax(hi) + span * 0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"{fam}: {name.lower()} ({unit}). Bottom 20% ≤ {res_by_key[(trait, outs[0])]['edges'][0]:.1f}, "
                 f"top 20% ≥ {res_by_key[(trait, outs[0])]['edges'][3]:.1f}", fontsize=11)
    fig.text(0.5, -0.02, "Pitch types with 100+ pitches, 2025–26. Dots: averages. Lines: 95% intervals (resampling pitchers).",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    # numpy on macOS (Accelerate) prints false matmul warnings; results are unaffected.
    with np.errstate(all="ignore"):
        _main()


def _main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(PROCESSED_DIR.glob("arsenal_*.parquet"))
    if not files:
        raise SystemExit("No processed data. Run python -m pitchlab.build 2025 2026 first.")
    a = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print("computing chase, batted-ball and hard-hit rates from raw data")
    a = prepare(a[a["n"] >= QUALIFY], contact_table())
    analysis_path = PROCESSED_DIR / "analysis.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    source = analysis.get("source", "unknown")

    rng = np.random.default_rng(2026)
    rows, ex = [], {}
    for fam, sub in a.groupby("family"):
        if sub["pitcher"].nunique() < 40:
            continue
        print(f"bootstrapping {fam} ({len(sub)} pitch types)")
        sub = sub.reset_index(drop=True)
        res = analyze_family(sub, fam, rng)
        rows += res
        by_key = {(r["trait"], r["outcome"]): r for r in res}
        for f2, t in HEADLINES:
            if f2 == fam and (t, "whiff_pct") in by_key:
                ex[f"{fam}|{t}"] = examples(sub, t)
                chart(fam, t, by_key, OUT / f"{fam.lower().replace('-', '')}_{t}.png")

    models = model_table(analysis)
    (OUT / "results.json").write_text(json.dumps(
        {"source": source, "seasons": sorted(int(s) for s in a["game_year"].unique()),
         "findings": rows, "examples": ex, "models": models}, indent=1, default=float))
    write_markdown(rows, ex, models, source)
    print(f"done: {OUT / 'results.md'}")


def write_markdown(rows, ex, models, source):
    L = [f"# Write-up results (data: {source})", "",
         "Top/bottom 20% = average for the pitch types (100+ pitches) highest and lowest on the trait.",
         "Per unit = change in the outcome for each 1 unit of the trait, with controls held fixed.",
         "[ ] = 95% interval from resampling pitchers. SOLID = gap and per-unit effect both clear of zero, same direction in both seasons.",
         "Whiff, chase and pop-up rates: higher is better for the pitcher. Hard-hit rate: lower is better. Ground-ball rate: neutral (depends on the pitcher's approach).",
         "Rates are percentages; gaps and per-unit effects are in percentage points.", ""]
    for fam in dict.fromkeys(r["family"] for r in rows):
        L += [f"## {fam}", "",
              "| Trait | Outcome | Bottom 20% → Top 20% | Gap [95%] | Per unit, controlled [95%] (controls) | Per unit 2025 / 2026 | Verdict |",
              "|---|---|---|---|---|---|---|"]
        for r in (x for x in rows if x["family"] == fam and x["outcome"] in MAIN_OUTCOMES):
            name, tunit = TRAITS[r["trait"]]
            _, _, ounit, _, _ = OUTCOMES[r["outcome"]]
            fmt = lambda v: f"{v:.1f}%"
            seas = " / ".join(f"{v:+.2f}" for v in r["per_unit_by_season"].values())
            L.append(f"| {name} ({tunit}) | {OUTCOMES[r['outcome']][0]} | {fmt(r['bottom20'])} → {fmt(r['top20'])} | "
                     f"{r['gap']:+.1f} [{r['gap_ci'][0]:+.1f}, {r['gap_ci'][1]:+.1f}] | "
                     f"{r['per_unit']:+.2f} {ounit} per {tunit} [{r['per_unit_ci'][0]:+.2f}, {r['per_unit_ci'][1]:+.2f}] "
                     f"({', '.join(r['controls'])}) | {seas} | **{r['verdict']}** ({r['direction']}) |")
        L.append("")

    L += ["## Appendix: xwOBA on contact and run value", "",
          "| Pitch | Trait | Outcome | Bottom 20% → Top 20% | Gap [95%] | Verdict |", "|---|---|---|---|---|---|"]
    for r in (x for x in rows if x["outcome"] in APPENDIX_OUTCOMES):
        if r["outcome"] == "xwoba_con":
            v = f".{int(round(r['bottom20'])):03d} → .{int(round(r['top20'])):03d}"
        else:
            v = f"{r['bottom20']:+.2f} → {r['top20']:+.2f}"
        L.append(f"| {r['family']} | {TRAITS[r['trait']][0]} | {OUTCOMES[r['outcome']][0]} | {v} | "
                 f"{r['gap']:+.2f} [{r['gap_ci'][0]:+.2f}, {r['gap_ci'][1]:+.2f}] | {r['verdict']} |")
    L += ["", "## Appendix: correlations (r) for analysts", "",
          "| Pitch | Trait | " + " | ".join(OUTCOMES[o][0] for o in OUTCOMES) + " |",
          "|---|---|" + "---|" * len(OUTCOMES)]
    by = {(r["family"], r["trait"], r["outcome"]): r["r"] for r in rows}
    for fam, t in dict.fromkeys((r["family"], r["trait"]) for r in rows):
        L.append(f"| {fam} | {TRAITS[t][0]} | " + " | ".join(
            f"{by[(fam, t, o)]:+.2f}" if (fam, t, o) in by else "–" for o in OUTCOMES) + " |")

    L += ["", "## Examples (300+ pitches, one season per pitcher)", ""]
    for key, e in ex.items():
        fam, t = key.split("|")
        L.append(f"**{fam}: {TRAITS[t][0].lower()} ({TRAITS[t][1]})**")
        for side, lab in (("top", "most"), ("bottom", "least")):
            L.append(f"- {lab}: " + "; ".join(
                f"{x['player_name']} {x['game_year']} {x['pitch_type']} ({x[t]:+.1f}, whiff {100 * x['whiff_pct']:.0f}%, "
                f"chase {100 * x.get('chase_pct', float('nan')):.0f}%, GB {100 * x.get('gb_pct', float('nan')):.0f}%, "
                f"hard-hit {100 * x.get('hardhit_pct', float('nan')):.0f}%, Stuff+ {x.get('stuff_plus', float('nan')):.0f})"
                for x in e[side]))
        L.append("")

    L += ["## Model validation (correlation with next season, 95% interval)", "",
          "| Pitch | Model | Next-season RV | Next-season whiff | Last season's results: RV / whiff |", "|---|---|---|---|---|"]
    f = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" if v else "–"
    for m in models:
        L.append(f"| {m['family']} | {m['model']} | {f(m.get('next_rv'))} | {f(m.get('next_whiff'))} | "
                 f"{f(m.get('next_rv_from_results'))} / {f(m.get('next_whiff_from_results'))} |")
    (OUT / "results.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
