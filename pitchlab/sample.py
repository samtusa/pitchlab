"""Synthetic Statcast-shaped data for testing and demos.

The relationships are rough approximations of real MLB patterns (higher arm
slots ride more, lower slots run more). Pitcher traits stay the same across
seasons so year-to-year checks work. Outcomes depend on pitch quality in a
direction that differs by pitch type (e.g. four-seamers are better with
extra ride, changeups with less). Three planted outliers let the tests check
the model finds them. Never mix this with real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# code: (ivb intercept, ivb per deg arm angle, hb intercept, hb per deg, velo mean)
_PROFILES = {
    "FF": (8.0, 0.18, 16.0, -0.20, 94.0),
    "SI": (1.0, 0.16, 20.0, -0.14, 93.5),
    "FC": (3.0, 0.12, -1.0, -0.03, 89.0),
    "SL": (-1.0, 0.07, -2.0, -0.05, 85.5),
    "ST": (-3.0, 0.07, -15.0, 0.06, 82.5),
    "CU": (-3.0, -0.22, -4.0, -0.05, 79.0),
    "CH": (-1.0, 0.16, 18.0, -0.12, 85.5),
}
# How residual movement (inches) turns into pitch quality, per pitch type:
# (weight on IVB residual, weight on arm-side HB residual)
_QUALITY = {
    "FF": (0.20, 0.0), "SI": (-0.05, 0.15), "FC": (0.0, -0.12), "SL": (-0.05, -0.15),
    "ST": (0.0, -0.18), "CU": (-0.18, 0.0), "CH": (-0.20, 0.08),
}
_ARSENALS = [
    ["FF", "SL", "CH"], ["SI", "SL", "CH"], ["FF", "CU", "CH"], ["FF", "ST", "SI"],
    ["SI", "ST", "CH"], ["FF", "FC", "CU"], ["FF", "SL", "CU", "CH"], ["SI", "FC", "ST"],
]
PLANTED = {
    # pitcher id: (pitch_type, extra IVB, extra arm-side HB)
    900001: ("FF", 6.0, 0.0),    # low slot, rides like an over-the-top arm
    900002: ("SI", 0.0, 7.0),    # extreme run for his slot
    900003: ("ST", 0.0, -8.0),   # sweeper far more glove-side than expected
}


def _traits(n_pitchers: int, seed: int):
    rng = np.random.default_rng(seed)
    ids = list(range(600000, 600000 + n_pitchers - len(PLANTED))) + list(PLANTED)
    out = []
    for i, pid in enumerate(ids):
        hand = "L" if rng.random() < 0.28 else "R"
        aa = 22.0 if pid == 900001 else float(np.clip(rng.normal(40, 12), -5, 70))
        arsenal = list(_ARSENALS[i % len(_ARSENALS)])
        if pid in PLANTED and PLANTED[pid][0] not in arsenal:
            arsenal.append(PLANTED[pid][0])
        pitches = {}
        for pt in arsenal:
            res_ivb, res_hb = rng.normal(0, 1.8), rng.normal(0, 1.8)
            if pid in PLANTED and PLANTED[pid][0] == pt:
                res_ivb += PLANTED[pid][1]
                res_hb += PLANTED[pid][2]
            pitches[pt] = {"res_ivb": res_ivb, "res_hb": res_hb, "velo_off": rng.normal(0, 0.8)}
        out.append({
            "pid": pid, "hand": hand, "aa": aa,
            "ext": float(np.clip(rng.normal(6.5, 0.45), 5.3, 7.9)),
            "velo_shift": rng.normal(0, 1.8),
            "usage": rng.dirichlet(np.ones(len(arsenal)) * 3),
            "pitches": pitches,
        })
    return out


def make_sample(season: int = 2025, n_pitchers: int = 420, seed: int = 7) -> pd.DataFrame:
    traits = _traits(n_pitchers, seed)
    rng = np.random.default_rng(seed * 10_000 + season)
    rows = []
    for t in traits:
        pid, hand, aa, ext = t["pid"], t["hand"], t["aa"], t["ext"]
        mirror = -1.0 if hand == "R" else 1.0
        rel_x_arm = 0.5 + 2.2 * np.cos(np.radians(aa))
        rel_z = 4.3 + 2.2 * np.sin(np.radians(aa))
        total = int(rng.integers(500, 3000))
        for (pt, tr), u in zip(t["pitches"].items(), t["usage"]):
            a_ivb, b_ivb, a_hb, b_hb, v0 = _PROFILES[pt]
            n = max(int(total * u), 5)
            if pid in PLANTED and PLANTED[pid][0] == pt:
                n = max(n, 400)
            velo = v0 + t["velo_shift"] + tr["velo_off"]
            ivb_mu = a_ivb + b_ivb * aa + 0.35 * (velo - v0) + tr["res_ivb"]
            hb_mu = a_hb + b_hb * aa + tr["res_hb"]
            wq_ivb, wq_hb = _QUALITY[pt]
            quality = wq_ivb * tr["res_ivb"] + wq_hb * tr["res_hb"] + 0.12 * (velo - v0)

            ivb = rng.normal(ivb_mu, 1.5, n)
            hb_arm = rng.normal(hb_mu, 1.5, n)
            speed = rng.normal(velo, 0.9, n)
            e = rng.normal(ext, 0.12, n)
            pfx_x, pfx_z = hb_arm / 12.0 * mirror, ivb / 12.0
            rz = rng.normal(rel_z, 0.08, n)
            plate_z = rng.normal(2.4, 0.9, n)

            # trajectory pieces (ft, ft/s, ft/s^2) consistent enough for VAA
            vy0 = -speed * 1.4667 * 0.995
            ay = np.full(n, 27.0)
            tf = (50 - 17 / 12) / (-vy0)
            az = -32.17 + 2 * pfx_z / tf ** 2
            z50 = rz - 0.3
            vz0 = (plate_z - z50 - 0.5 * az * tf ** 2) / tf

            # outcomes; breaking balls are nastier against same-side hitters
            stand = np.where(rng.random(n) < 0.55, "R", "L")
            same = (stand == hand).astype(float)
            q_pitch = quality + (0.35 * same if pt in ("SL", "ST", "CU") else 0.0)
            in_zone = rng.random(n) < 0.48
            zone = np.where(in_zone, rng.integers(1, 10, n), rng.integers(11, 15, n))
            swing_p = np.where(in_zone, 0.66, np.clip(0.28 + 0.05 * q_pitch, 0.05, 0.8))
            swing = rng.random(n) < swing_p
            whiff_p = 1 / (1 + np.exp(-(-1.55 + 0.9 * q_pitch)))
            whiff = swing & (rng.random(n) < whiff_p)
            contact = swing & ~whiff
            in_play = contact & (rng.random(n) < 0.45)
            called = ~swing & (rng.random(n) < 0.33)
            desc = np.where(whiff, "swinging_strike", np.where(in_play, "hit_into_play",
                   np.where(contact, "foul", np.where(called, "called_strike", "ball"))))
            xwoba = np.where(in_play, np.clip(rng.normal(0.37 - 0.08 * q_pitch, 0.25, n), 0, 2), np.nan)
            # batted-ball type: more ride -> more fly balls and pop-ups, more drop -> grounders
            gb_p = np.clip(0.44 - 0.012 * ivb, 0.15, 0.75)
            pu_p = np.clip(0.06 + 0.004 * ivb, 0.01, 0.2)
            u = rng.random(n)
            bb_type = np.where(~in_play, None, np.where(u < gb_p, "ground_ball", np.where(
                u < gb_p + pu_p, "popup", np.where(u < gb_p + pu_p + 0.22, "line_drive", "fly_ball"))))
            launch_speed = np.where(in_play, rng.normal(89 - 2.0 * q_pitch, 13, n), np.nan)
            dre = np.select(
                [whiff | called, desc == "foul", desc == "ball", in_play],
                [-0.06, -0.035, 0.055, (xwoba - 0.32) * 1.1],
            ) + rng.normal(0, 0.02, n)
            dre = np.where(in_play, dre + rng.normal(0, 0.25, n), dre)

            rows.append(pd.DataFrame({
                "game_date": pd.Timestamp(f"{season}-06-01"),
                "game_year": season,
                "pitcher": pid,
                "player_name": f"P{pid}, Sample",
                "p_throws": hand,
                "stand": stand,
                "pitch_type": pt,
                "release_speed": speed,
                "effective_speed": speed * (60.5 - 6.5) / (60.5 - e),
                "release_spin_rate": rng.normal(2300, 150, n),
                "spin_axis": (180 - np.degrees(np.arctan2(pfx_x, pfx_z))) % 360,
                "release_extension": e,
                "release_pos_x": rng.normal(rel_x_arm * mirror, 0.08, n),
                "release_pos_y": 60.5 - e,
                "release_pos_z": rz,
                "pfx_x": pfx_x,
                "pfx_z": pfx_z,
                "plate_x": rng.normal(0, 0.8, n),
                "plate_z": plate_z,
                "vy0": vy0, "vz0": vz0, "ay": ay, "az": az,
                "arm_angle": rng.normal(aa, 1.5, n),
                "game_type": "R",
                "description": desc,
                "delta_run_exp": dre,
                "estimated_woba_using_speedangle": xwoba,
                "zone": zone,
                "bb_type": bb_type,
                "launch_speed": launch_speed,
            }))
    return pd.concat(rows, ignore_index=True)
