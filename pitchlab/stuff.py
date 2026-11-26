"""PitchLab Stuff+ and the "does slot-adjusted movement matter?" analysis.

Target
------
Run value from the pitcher's side (-delta_run_exp). Balls in play are
de-noised: their run value is replaced by the league-average run value for
contact of that quality (xwOBA bin), so a bloop single and a lineout with the
same contact quality count the same. Location is NOT a feature, so the model
grades the pitch's physical traits only.

Validation
----------
Models are fit separately per pitch family with 5-fold cross-validation
grouped by pitcher: every prediction comes from a model that never saw that
pitcher in any season. Three feature sets are compared:

  movement only   velo, IVB, HB, spin, extension, approach angle,
                  separation from the pitcher's fastball
  + release       adds release side and height
  + slot          adds arm angle and slot-adjusted movement (IVB+/HB+)

If "+ slot" beats "+ release" out of sample, slot-relative movement carries
information beyond raw movement and release point.

Version 2 (the Stuff+ shown on the site)
----------------------------------------
- adds the batter's handedness (same side vs opposite side); grades are then
  blended at the league platoon mix so nobody is credited for his matchups
- trains sliders and sweepers together (with a sweeper flag), since the
  Statcast label boundary between them is fuzzy and each alone is small
- instead of one noisy run-value target, fits separate models for swing,
  whiff per swing, ball in play per contact, called strike per take, and
  contact quality, then combines them with league run values per outcome
- stronger regularization (bigger leaves) for every model
The v1 variants above stay in the report for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

QUALIFY = 100  # pitches of a type before it counts in evaluation / Stuff+ scale
MAX_TRAIN = 400_000  # rows per fold; keeps runtime reasonable on a laptop

BASE = ["release_speed", "ivb_in", "hb_arm_in", "release_spin_rate", "release_extension",
        "vaa_adj", "d_velo", "d_ivb", "d_hb"]
RELEASE = ["release_x_arm", "release_z"]
SLOT = ["arm_angle", "ivb_oe_p", "hb_oe_p"]
VARIANTS = {
    "movement only": BASE,
    "+ release point": BASE + RELEASE,
    "+ arm slot & slot-adjusted movement": BASE + RELEASE + SLOT,
}
V1_FINAL = "+ arm slot & slot-adjusted movement"
V2_NAME = "v2: + batter hand, outcome models, SL/ST pooled"
V2_FEATURES = BASE + RELEASE + SLOT + ["same_side", "is_sweeper"]
# Pitch families trained together in v2
V2_GROUPS = {"Breaking ball": ["Slider", "Sweeper"]}
V2_MAX_TRAIN = 300_000


def has_outcomes(p: pd.DataFrame) -> bool:
    return "delta_run_exp" in p and p["delta_run_exp"].notna().mean() > 0.5


def make_target(p: pd.DataFrame) -> pd.Series:
    rv = -p["delta_run_exp"].astype(float)
    xw = p.get("estimated_woba_using_speedangle")
    if xw is not None:
        inplay = p["in_play"] & xw.notna() & rv.notna()
        bins = (xw[inplay].clip(0, 2) / 0.05).round()
        rv.loc[inplay] = bins.map(rv[inplay].groupby(bins).mean())
    return rv


def add_vaa_adj(p: pd.DataFrame) -> None:
    """Approach angle with pitch height removed (high pitches are naturally flatter)."""
    p["vaa_adj"] = np.nan
    for fam, idx in p.groupby("family").groups.items():
        sub = p.loc[idx]
        ok = sub["vaa"].notna() & sub.get("plate_z", pd.Series(np.nan, index=sub.index)).notna()
        if ok.sum() > 100:
            b, a = np.polyfit(sub.loc[ok, "plate_z"], sub.loc[ok, "vaa"], 1)
            p.loc[idx, "vaa_adj"] = sub["vaa"] - (a + b * sub["plate_z"])
        else:
            p.loc[idx, "vaa_adj"] = sub["vaa"] - sub["vaa"].mean()


def _model():
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=300, l2_regularization=1.0, random_state=0)


def _oof_predict(df: pd.DataFrame, feats: list[str], folds: int = 5) -> np.ndarray:
    X = df[feats].to_numpy(float)
    y = df["target"].to_numpy(float)
    groups = df["pitcher"].to_numpy()
    pred = np.full(len(df), np.nan)
    rng = np.random.default_rng(0)
    k = min(folds, len(np.unique(groups)))
    for tr, te in GroupKFold(k).split(X, y, groups):
        if len(tr) > MAX_TRAIN:
            tr = rng.choice(tr, MAX_TRAIN, replace=False)
        m = _model().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def _wcorr(x, y, w) -> float:
    x, y, w = (np.asarray(a, float) for a in (x, y, w))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if ok.sum() < 10:
        return float("nan")
    x, y, w = x[ok], y[ok], w[ok]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return float(cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)))


def _group_eval(p: pd.DataFrame, pred_col: str, whiff_col: str | None = None) -> dict:
    """Correlate a pitch type's average predicted value with its actual results."""
    keys = ["game_year", "pitcher", "pitch_type"]
    wcol = whiff_col or pred_col
    g = (p.assign(_rv=-p["delta_run_exp"], _pw=p[wcol])
         .groupby(keys)
         .agg(n=("_rv", "size"), pred=(pred_col, "mean"), rv=("_rv", "mean"), pw=("_pw", "mean"),
              swings=("swing", "sum"), whiffs=("whiff", "sum"))
         .reset_index())
    g = g[g["n"] >= QUALIFY]
    g["whiff"] = g["whiffs"] / g["swings"].where(g["swings"] > 0)
    out = {
        "same_rv": _wcorr(g["pred"], g["rv"], g["n"]),
        "same_whiff": _wcorr(g["pw"], g["whiff"], g["swings"]),
        "rows": int(len(g)),
    }
    # Predictiveness: this season's grade vs next season's results, compared
    # with simply using this season's results.
    nxt = g.assign(game_year=g["game_year"] - 1)
    pair = g.merge(nxt, on=["game_year", "pitcher", "pitch_type"], suffixes=("", "_next"))
    if len(pair) >= 20:
        w = np.minimum(pair["n"], pair["n_next"])
        out.update({
            "next_rv": _wcorr(pair["pred"], pair["rv_next"], w),
            "next_rv_from_results": _wcorr(pair["rv"], pair["rv_next"], w),
            "next_whiff": _wcorr(pair["pw"], pair["whiff_next"], w),
            "next_whiff_from_results": _wcorr(pair["whiff"], pair["whiff_next"], w),
            "stability": _wcorr(pair["pred"], pair["pred_next"], w),
            "pairs": int(len(pair)),
        })
    return {k: (round(v, 3) if isinstance(v, float) and np.isfinite(v) else v) for k, v in out.items()}


def fit_stuff(pitches: pd.DataFrame, families=None, log=print):
    """Return (pitches with stuff_pred column, validation report dict)."""
    p = pitches.copy()
    p["target"] = make_target(p)
    add_vaa_adj(p)
    p["stuff_pred"] = np.nan
    report = {}
    fams = families or sorted(p["family"].dropna().unique())
    for fam in fams:
        idx = p.index[(p["family"] == fam) & p["target"].notna()]
        sub = p.loc[idx]
        if sub["pitcher"].nunique() < 30:
            continue
        report[fam] = {}
        for name, feats in VARIANTS.items():
            log(f"  {fam}: v1 {name}")
            col = f"_pred_{len(feats)}"
            sub = sub.assign(**{col: _oof_predict(sub, feats)})
            report[fam][name] = _group_eval(sub, col)

    # Version 2 is what the site shows as Stuff+.
    p = fit_stuff_v2(p, log=log)
    p["stuff_pred"] = p["v2_pred"]
    for fam in report:
        sub = p[(p["family"] == fam) & p["target"].notna() & p["v2_pred"].notna()]
        if len(sub):
            report[fam][V2_NAME] = _group_eval(sub, "v2_pred", "v2_whiff")
    return p, report


def _clf():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=600, l2_regularization=2.0, random_state=0)


def _reg():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=600, l2_regularization=2.0, random_state=0)


def event_values(p: pd.DataFrame) -> dict:
    """League-average run value (pitcher's side) of each pitch outcome."""
    rv = -p["delta_run_exp"]
    contact_foul = p["swing"] & ~p["whiff"] & ~p["in_play"]
    take = ~p["swing"]
    return {
        "whiff": float(rv[p["whiff"]].mean()),
        "foul": float(rv[contact_foul].mean()),
        "called_strike": float(rv[p["called_strike"]].mean()),
        "ball": float(rv[take & ~p["called_strike"]].mean()),
    }


def _v2_oof(df: pd.DataFrame, ev: dict, folds: int = 5):
    """Out-of-fold expected run value and whiff-per-swing for one model group.

    Predictions are made for a same-side hitter and an opposite-side hitter
    and blended at the league's platoon mix, so a pitcher is not graded up or
    down for the batters he happened to face.
    """
    X = df[V2_FEATURES].to_numpy(float)
    side_col = V2_FEATURES.index("same_side")
    X_opp, X_same = X.copy(), X.copy()
    X_opp[:, side_col], X_same[:, side_col] = 0.0, 1.0
    share_same = float(np.nanmean(X[:, side_col])) if np.isfinite(X[:, side_col]).any() else 0.5
    groups = df["pitcher"].to_numpy()
    swing = df["swing"].to_numpy()
    whiff = df["whiff"].to_numpy()
    in_play = df["in_play"].to_numpy()
    called = df["called_strike"].to_numpy()
    target = df["target"].to_numpy(float)
    keys = ("swing", "whiff", "in_play", "called", "bip")
    out = {side: {k: np.full(len(df), np.nan) for k in keys} for side in ("opp", "same")}
    rng = np.random.default_rng(1)
    k = min(folds, len(np.unique(groups)))

    def cap(idx):
        return rng.choice(idx, V2_MAX_TRAIN, replace=False) if len(idx) > V2_MAX_TRAIN else idx

    for tr, te in GroupKFold(k).split(X, groups=groups):
        sw = tr[swing[tr]]
        contact = sw[~whiff[sw]]
        takes = tr[~swing[tr]]
        bip = tr[in_play[tr] & np.isfinite(target[tr])]
        for key, idx, y in (("swing", tr, swing), ("whiff", sw, whiff),
                            ("in_play", contact, in_play), ("called", takes, called)):
            idx = cap(idx)
            m = _clf().fit(X[idx], y[idx])
            out["opp"][key][te] = m.predict_proba(X_opp[te])[:, 1]
            out["same"][key][te] = m.predict_proba(X_same[te])[:, 1]
        idx = cap(bip)
        m = _reg().fit(X[idx], target[idx])
        out["opp"]["bip"][te] = m.predict(X_opp[te])
        out["same"]["bip"][te] = m.predict(X_same[te])

    def combine(o):
        ps, pw, pip, pcs = o["swing"], o["whiff"], o["in_play"], o["called"]
        return (ps * (pw * ev["whiff"] + (1 - pw) * (pip * o["bip"] + (1 - pip) * ev["foul"]))
                + (1 - ps) * (pcs * ev["called_strike"] + (1 - pcs) * ev["ball"]))

    exp_rv = share_same * combine(out["same"]) + (1 - share_same) * combine(out["opp"])
    pw = share_same * out["same"]["whiff"] + (1 - share_same) * out["opp"]["whiff"]
    return exp_rv, pw


def fit_stuff_v2(p: pd.DataFrame, log=print) -> pd.DataFrame:
    """Adds v2_pred (expected run value) and v2_whiff (whiff per swing) to p."""
    p["same_side"] = (p["stand"] == p["p_throws"]).astype(float) if "stand" in p else np.nan
    p["is_sweeper"] = (p["pitch_type"] == "ST").astype(float)
    p["v2_pred"] = np.nan
    p["v2_whiff"] = np.nan
    ev = event_values(p)
    fam_to_group = {f: g for g, fs in V2_GROUPS.items() for f in fs}
    p["_group"] = p["family"].map(lambda f: fam_to_group.get(f, f))
    for grp, idx in p.groupby("_group").groups.items():
        sub = p.loc[idx]
        if sub["pitcher"].nunique() < 30:
            continue
        log(f"  {grp}: version 2 outcome models")
        rv, pw = _v2_oof(sub, ev)
        p.loc[idx, "v2_pred"] = rv
        p.loc[idx, "v2_whiff"] = pw
    return p.drop(columns="_group")


def stuff_plus_table(p: pd.DataFrame) -> pd.DataFrame:
    """Stuff+ per season x pitcher x pitch type. 100 = league average pitch,
    10 points = one standard deviation among qualified pitch types."""
    keys = ["game_year", "pitcher", "pitch_type"]
    out = []
    for season, sp in p.dropna(subset=["stuff_pred"]).groupby("game_year"):
        lg = sp["stuff_pred"].mean()
        g = sp.groupby(keys).agg(n=("stuff_pred", "size"), pred=("stuff_pred", "mean"),
                                 vaa_adj=("vaa_adj", "mean")).reset_index()
        q = g[g["n"] >= QUALIFY]
        sd = np.sqrt(np.average((q["pred"] - lg) ** 2, weights=np.sqrt(q["n"]))) if len(q) > 5 else g["pred"].std()
        g["stuff_plus"] = 100 + 10 * (g["pred"] - lg) / sd
        out.append(g[keys + ["stuff_plus", "vaa_adj"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=keys + ["stuff_plus", "vaa_adj"])


def quintile_analysis(arsenal: pd.DataFrame) -> dict:
    """Do pitches with more slot-adjusted movement get better results?

    Pools all seasons. For each family, splits qualified pitch types into
    quintiles of IVB+ and of HB+ and reports actual results per quintile."""
    a = arsenal[(arsenal["n"] >= QUALIFY) & arsenal["ivb_oe"].notna()]
    if "rv100" not in a:
        return {}
    res = {}
    for fam, sub in a.groupby("family"):
        if len(sub) < 25:
            continue
        fam_out = {"pitch_types": int(len(sub))}
        extra = ["vaa_adj"] if fam in ("Four-Seam", "Sinker") else ["d_velo", "d_ivb", "d_hb"]
        for col in ["ivb_oe", "hb_oe"] + [c for c in extra if c in sub and sub[c].notna().sum() >= 25]:
            q = pd.qcut(sub[col], 5, labels=False, duplicates="drop")
            bins = []
            for k, b in sub.groupby(q):
                bins.append({
                    "quintile": int(k) + 1,
                    "lo": round(float(b[col].min()), 1), "hi": round(float(b[col].max()), 1),
                    "count": int(len(b)),
                    "rv100": round(float(np.average(b["rv100"], weights=b["n"])), 2),
                    "whiff_pct": round(float(b["whiffs"].sum() / max(b["swings"].sum(), 1)), 3),
                    "stuff_plus": round(float(b["stuff_plus"].mean()), 1) if "stuff_plus" in b and b["stuff_plus"].notna().any() else None,
                })
            fam_out[col] = {
                "bins": bins,
                "corr_rv100": round(_wcorr(sub[col], sub["rv100"], sub["n"]), 3),
                "corr_whiff": round(_wcorr(sub[col], sub["whiff_pct"], sub["swings"]), 3),
                "corr_stuff": round(_wcorr(sub[col], sub["stuff_plus"], sub["n"]), 3) if "stuff_plus" in sub else None,
            }
        res[fam] = fam_out
    return res
