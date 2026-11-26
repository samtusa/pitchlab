"""Pitch-level derived features.

Conventions (all from the catcher's view in raw Statcast):
- pfx_x / pfx_z are induced movement in feet (gravity removed).
- release_pos_x is negative for most right-handers (3B side).

We mirror left-handers so every number reads from the pitcher's arm side:
- hb_arm_in > 0  -> moves toward the pitcher's arm side (run)
- hb_arm_in < 0  -> moves toward the glove side (sweep/cut)
- release_x_arm  -> how far toward the arm side the ball is released
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MOUND_DISTANCE_FT, PITCH_FAMILIES

# Fallback shoulder height (ft) used only when Savant's arm_angle is missing.
_FALLBACK_SHOULDER_FT = 4.3


def spin_axis_to_tilt(axis_deg: pd.Series) -> pd.Series:
    """Convert Savant spin axis (180 = pure backspin) to clock tilt, e.g. '1:15'."""
    hours = ((axis_deg / 30.0) + 6.0) % 12.0
    minutes = (np.round(hours * 4) / 4 * 60).astype("Int64")  # nearest 15 min

    def fmt(m):
        if pd.isna(m):
            return None
        h, mm = divmod(int(m), 60)
        h = h % 12 or 12
        return f"{h}:{mm:02d}"

    return minutes.map(fmt)


def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df[df["pitch_type"].isin(PITCH_FAMILIES)]
    df = df.dropna(subset=["pfx_x", "pfx_z", "release_speed", "release_pos_x", "release_pos_z", "p_throws"])
    if "game_type" in df:
        df = df[df["game_type"].fillna("R") == "R"]

    mirror = np.where(df["p_throws"] == "R", -1.0, 1.0)
    df["family"] = df["pitch_type"].map(PITCH_FAMILIES)
    df["ivb_in"] = df["pfx_z"] * 12.0
    df["hb_arm_in"] = df["pfx_x"] * 12.0 * mirror
    df["release_x_arm"] = df["release_pos_x"] * mirror
    df["release_z"] = df["release_pos_z"]

    # Arm angle: use Savant's value; estimate only where it is missing.
    if "arm_angle" not in df:
        df["arm_angle"] = np.nan
    est = np.degrees(np.arctan2(df["release_z"] - _FALLBACK_SHOULDER_FT,
                                df["release_x_arm"].clip(lower=0.1)))
    df["arm_angle_estimated"] = df["arm_angle"].isna()
    df["arm_angle"] = df["arm_angle"].fillna(est)

    # Perceived velocity: scale release speed by how much closer to the plate
    # this pitcher lets go than a league-average extension.
    if "release_extension" not in df:
        df["release_extension"] = np.nan
    lg_ext = df["release_extension"].median()
    ext = df["release_extension"].fillna(lg_ext)
    df["perceived_velo"] = df["release_speed"] * (MOUND_DISTANCE_FT - lg_ext) / (MOUND_DISTANCE_FT - ext)
    df.attrs["league_extension_ft"] = float(lg_ext)

    if "spin_axis" in df:
        df["spin_tilt"] = spin_axis_to_tilt(df["spin_axis"])
    df["vaa"] = vertical_approach_angle(df)
    add_outcome_flags(df)
    add_fastball_separation(df)
    return df


def vertical_approach_angle(df: pd.DataFrame) -> pd.Series:
    """Angle (deg) the pitch crosses the front of the plate; more negative = steeper.

    Uses Statcast's trajectory fit, which starts at y = 50 ft.
    """
    need = ["vy0", "vz0", "ay", "az"]
    if not all(c in df for c in need):
        return pd.Series(np.nan, index=df.index)
    y_front = 17.0 / 12.0
    vy_f = -np.sqrt(np.clip(df["vy0"] ** 2 - 2 * df["ay"] * (50.0 - y_front), 0, None))
    t = (vy_f - df["vy0"]) / df["ay"]
    vz_f = df["vz0"] + df["az"] * t
    return -np.degrees(np.arctan(vz_f / vy_f))


SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
          "foul_bunt", "missed_bunt", "bunt_foul_tip"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}


def add_outcome_flags(df: pd.DataFrame) -> None:
    if "description" not in df:
        return
    d = df["description"]
    df["swing"] = d.isin(SWINGS)
    df["whiff"] = d.isin(WHIFFS)
    df["called_strike"] = d.eq("called_strike")
    df["in_play"] = d.eq("hit_into_play")


def add_fastball_separation(df: pd.DataFrame) -> None:
    """Velo and movement differences from the pitcher's primary fastball.

    Primary fastball = his most-used four-seam or sinker (cutter if he has
    neither). For the fastball itself the differences are 0.
    """
    keys = ["game_year", "pitcher"]
    fb = df[df["pitch_type"].isin(["FF", "SI", "FC"])]
    counts = fb.groupby(keys + ["pitch_type"]).size().rename("cnt").reset_index()
    counts["pref"] = counts["pitch_type"].map({"FF": 0, "SI": 0, "FC": 1})
    primary = (counts.sort_values(["pref", "cnt"], ascending=[True, False])
               .drop_duplicates(keys)[keys + ["pitch_type"]].rename(columns={"pitch_type": "fb_type"}))
    fbm = (fb.merge(primary, on=keys)
           .query("pitch_type == fb_type")
           .groupby(keys)
           .agg(fb_type=("fb_type", "first"), fb_velo=("release_speed", "mean"),
                fb_ivb=("ivb_in", "mean"), fb_hb=("hb_arm_in", "mean"))
           .reset_index())
    m = df[keys].merge(fbm, on=keys, how="left")
    m.index = df.index
    df["d_velo"] = df["release_speed"] - m["fb_velo"]
    df["d_ivb"] = df["ivb_in"] - m["fb_ivb"]
    df["d_hb"] = df["hb_arm_in"] - m["fb_hb"]
    same = df["pitch_type"].eq(m["fb_type"])
    df.loc[same, ["d_velo", "d_ivb", "d_hb"]] = 0.0
