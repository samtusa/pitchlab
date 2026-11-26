"""Pitcher arsenal aggregation and the slot-adjusted movement model.

Movement over expected (MOE)
----------------------------
For each pitch family (four-seam, sinker, slider, ...) we fit, across every
qualifying pitcher, the expected induced vertical break (IVB) and arm-side
horizontal break (HB) as a function of:
    arm angle (quadratic) + release speed (linear)
weighted by pitch count. A pitcher's MOE is actual minus expected, shrunk
toward zero for small samples. The slot outlier score is the Mahalanobis
distance of his (IVB, HB) residual from the league residual distribution, so
2.0 means "about as unusual as a 2-sigma outlier" in 2D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_PITCHES, SHRINK_K

MIN_PITCHERS_PER_FAMILY = 15
WEIGHT_CAP = 600  # one workhorse cannot dominate the fit


def _circular_mean_deg(a: pd.Series) -> float:
    a = a.dropna()
    if a.empty:
        return np.nan
    r = np.radians(a)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def aggregate_arsenal(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per season x pitcher x pitch type."""
    keys = ["game_year", "pitcher", "pitch_type"]
    g = pitches.groupby(keys, observed=True)
    out = g.agg(
        player_name=("player_name", "first"),
        p_throws=("p_throws", "first"),
        family=("family", "first"),
        n=("pfx_x", "size"),
        velo=("release_speed", "mean"),
        perceived_velo=("perceived_velo", "mean"),
        effective_speed=("effective_speed", "mean"),
        extension=("release_extension", "mean"),
        arm_angle=("arm_angle", "mean"),
        arm_angle_estimated=("arm_angle_estimated", "mean"),
        release_x_arm=("release_x_arm", "mean"),
        release_z=("release_z", "mean"),
        ivb=("ivb_in", "mean"),
        hb=("hb_arm_in", "mean"),
        spin_rate=("release_spin_rate", "mean"),
    ).reset_index()
    extra = {
        "vaa": ("vaa", "mean"), "d_velo": ("d_velo", "mean"),
        "d_ivb": ("d_ivb", "mean"), "d_hb": ("d_hb", "mean"),
    }
    if "delta_run_exp" in pitches and pitches["delta_run_exp"].notna().any():
        p2 = pitches.assign(
            _rv=-pitches["delta_run_exp"],
            _csw=pitches["whiff"] | pitches["called_strike"],
            _xw=pitches["estimated_woba_using_speedangle"].where(pitches["in_play"]),
        )
        g = p2.groupby(keys, observed=True)
        extra.update({
            "rv": ("_rv", "sum"), "rv_n": ("_rv", "count"), "swings": ("swing", "sum"),
            "whiffs": ("whiff", "sum"), "csw": ("_csw", "mean"), "xwoba_con": ("_xw", "mean"),
        })
    ext = g.agg(**{k: v for k, v in extra.items() if v[0] in g.obj}).reset_index()
    out = out.merge(ext, on=keys, how="left")
    if "rv" in out:
        out["rv100"] = 100 * out["rv"] / out["rv_n"].where(out["rv_n"] > 0)
        out["whiff_pct"] = out["whiffs"] / out["swings"].where(out["swings"] > 0)
        out = out.drop(columns=["rv", "rv_n"])
    if "spin_axis" in pitches:
        axis = pitches.groupby(keys, observed=True)["spin_axis"].apply(_circular_mean_deg).rename("spin_axis").reset_index()
        out = out.merge(axis, on=keys, how="left")
    totals = out.groupby(["game_year", "pitcher"])["n"].transform("sum")
    out["usage"] = out["n"] / totals
    out["velo_gain"] = out["perceived_velo"] - out["velo"]
    return out


def _design(arm_angle, velo, aa_mu, v_mu):
    a = (np.asarray(arm_angle) - aa_mu) / 10.0
    v = np.asarray(velo) - v_mu
    return np.column_stack([np.ones_like(a), a, a ** 2, v])


def _wls(X, y, w, ridge=1e-3):
    W = np.sqrt(w)[:, None]
    Xw, yw = X * W, y * W[:, 0]
    reg = ridge * np.eye(X.shape[1])
    reg[0, 0] = 0
    return np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ yw)


def fit_movement_models(arsenal: pd.DataFrame):
    """Return (arsenal with MOE columns, per-family model curves for charts)."""
    # numpy 2.0 on macOS (Accelerate BLAS) emits spurious matmul warnings;
    # real problems are caught by the finite checks below instead.
    with np.errstate(all="ignore"):
        return _fit_movement_models(arsenal)


def _fit_movement_models(arsenal: pd.DataFrame):
    df = arsenal.copy()
    for c in ["exp_ivb", "exp_hb", "ivb_oe", "hb_oe", "ivb_oe_z", "hb_oe_z", "slot_outlier"]:
        df[c] = np.nan
    curves = {}

    for (season, fam), sub in df.groupby(["game_year", "family"]):
        fit_rows = sub[(sub["n"] >= MIN_PITCHES) & sub["arm_angle"].notna()]
        if len(fit_rows) < MIN_PITCHERS_PER_FAMILY:
            continue
        w = fit_rows["n"].clip(upper=WEIGHT_CAP).to_numpy(float)
        aa_mu = float(np.average(fit_rows["arm_angle"], weights=w))
        v_mu = float(np.average(fit_rows["velo"], weights=w))
        X = _design(fit_rows["arm_angle"], fit_rows["velo"], aa_mu, v_mu)
        b_ivb = _wls(X, fit_rows["ivb"].to_numpy(float), w)
        b_hb = _wls(X, fit_rows["hb"].to_numpy(float), w)
        if not (np.isfinite(b_ivb).all() and np.isfinite(b_hb).all()):
            print(f"warning: {season} {fam} model did not fit; skipped")
            continue

        # Residual spread among qualifiers sets the z-score scale.
        res = np.column_stack([fit_rows["ivb"] - X @ b_ivb, fit_rows["hb"] - X @ b_hb])
        cov = np.cov(res.T, aweights=w)
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(2))
        sd_ivb, sd_hb = np.sqrt(np.diag(cov))

        idx = sub.index[sub["arm_angle"].notna()]
        Xa = _design(df.loc[idx, "arm_angle"], df.loc[idx, "velo"], aa_mu, v_mu)
        exp_ivb, exp_hb = Xa @ b_ivb, Xa @ b_hb
        shrink = df.loc[idx, "n"] / (df.loc[idx, "n"] + SHRINK_K)
        ivb_oe = (df.loc[idx, "ivb"] - exp_ivb) * shrink
        hb_oe = (df.loc[idx, "hb"] - exp_hb) * shrink
        r = np.column_stack([ivb_oe, hb_oe])
        maha = np.sqrt(np.einsum("ij,jk,ik->i", r, cov_inv, r))

        df.loc[idx, "exp_ivb"] = exp_ivb
        df.loc[idx, "exp_hb"] = exp_hb
        df.loc[idx, "ivb_oe"] = ivb_oe
        df.loc[idx, "hb_oe"] = hb_oe
        df.loc[idx, "ivb_oe_z"] = ivb_oe / sd_ivb
        df.loc[idx, "hb_oe_z"] = hb_oe / sd_hb
        df.loc[idx, "slot_outlier"] = maha

        lo, hi = np.percentile(fit_rows["arm_angle"], [2, 98])
        grid = np.linspace(lo, hi, 40)
        Xg = _design(grid, np.full_like(grid, v_mu), aa_mu, v_mu)
        curves.setdefault(int(season), {})[fam] = {
            "arm_angle": grid.round(1).tolist(),
            "exp_ivb": (Xg @ b_ivb).round(2).tolist(),
            "exp_hb": (Xg @ b_hb).round(2).tolist(),
            "league_velo": round(v_mu, 1),
            "sd_ivb": round(float(sd_ivb), 2),
            "sd_hb": round(float(sd_hb), 2),
            "n_pitchers": int(len(fit_rows)),
            "r2_ivb": round(_r2(fit_rows["ivb"], X @ b_ivb, w), 3),
            "r2_hb": round(_r2(fit_rows["hb"], X @ b_hb, w), 3),
            # raw coefficients, so pitch-level residuals can be computed later
            "aa_mu": aa_mu, "v_mu": v_mu,
            "coef_ivb": b_ivb.tolist(), "coef_hb": b_hb.tolist(),
        }
    return df, curves


def add_pitch_moe(pitches: pd.DataFrame, curves: dict) -> pd.DataFrame:
    """Unshrunk slot residuals for every pitch, using the season's family models."""
    p = pitches
    p["ivb_oe_p"] = np.nan
    p["hb_oe_p"] = np.nan
    with np.errstate(all="ignore"):
        for (season, fam), idx in p.groupby(["game_year", "family"]).groups.items():
            c = curves.get(int(season), {}).get(fam)
            if not c:
                continue
            X = _design(p.loc[idx, "arm_angle"], p.loc[idx, "release_speed"], c["aa_mu"], c["v_mu"])
            p.loc[idx, "ivb_oe_p"] = p.loc[idx, "ivb_in"].to_numpy() - X @ np.array(c["coef_ivb"])
            p.loc[idx, "hb_oe_p"] = p.loc[idx, "hb_arm_in"].to_numpy() - X @ np.array(c["coef_hb"])
    return p


def _r2(y, yhat, w):
    y = np.asarray(y, float)
    mu = np.average(y, weights=w)
    return float(1 - np.sum(w * (y - yhat) ** 2) / np.sum(w * (y - mu) ** 2))


def _wavg(sub: pd.DataFrame, col: str) -> float:
    if col not in sub:
        return np.nan
    ok = sub[col].notna()
    return float(np.average(sub.loc[ok, col], weights=sub.loc[ok, "n"])) if ok.any() else np.nan


def pitcher_summary(arsenal: pd.DataFrame) -> pd.DataFrame:
    """One row per season x pitcher: slot, extension, primary fastball velo."""
    rows = []
    for (season, pid), sub in arsenal.groupby(["game_year", "pitcher"]):
        w = sub["n"]
        fb = sub[sub["family"].isin(["Four-Seam", "Sinker"])].sort_values("n", ascending=False)
        top = sub.sort_values("slot_outlier", ascending=False).dropna(subset=["slot_outlier"])
        top = top[top["n"] >= MIN_PITCHES]
        rows.append({
            "game_year": season,
            "pitcher": pid,
            "player_name": sub["player_name"].iloc[0],
            "p_throws": sub["p_throws"].iloc[0],
            "pitches": int(w.sum()),
            "arm_angle": float(np.average(sub["arm_angle"], weights=w)),
            "extension": float(np.average(sub["extension"].fillna(sub["extension"].mean()), weights=w)),
            "fb_type": fb["pitch_type"].iloc[0] if len(fb) else None,
            "fb_velo": float(fb["velo"].iloc[0]) if len(fb) else np.nan,
            "fb_perceived": float(fb["perceived_velo"].iloc[0]) if len(fb) else np.nan,
            "top_outlier_pitch": top["pitch_type"].iloc[0] if len(top) else None,
            "top_outlier_score": float(top["slot_outlier"].iloc[0]) if len(top) else np.nan,
            "stuff_plus": _wavg(sub, "stuff_plus"),
        })
    return pd.DataFrame(rows)
