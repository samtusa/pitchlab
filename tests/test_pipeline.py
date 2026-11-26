import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pitchlab.features import add_features, spin_axis_to_tilt
from pitchlab.metrics import aggregate_arsenal, fit_movement_models
from pitchlab.sample import PLANTED, make_sample


@pytest.fixture(scope="module")
def built():
    pitches = add_features(make_sample(2025, n_pitchers=300, seed=3))
    arsenal, curves = fit_movement_models(aggregate_arsenal(pitches))
    return pitches, arsenal, curves


def test_lefties_are_mirrored(built):
    pitches, _, _ = built
    ff = pitches[pitches.pitch_type == "FF"]
    # Four-seamers run to the arm side for both hands after mirroring.
    assert ff[ff.p_throws == "R"].hb_arm_in.mean() > 4
    assert ff[ff.p_throws == "L"].hb_arm_in.mean() > 4


def test_perceived_velo_direction(built):
    pitches, _, _ = built
    long_ext = pitches[pitches.release_extension > 7.2]
    short_ext = pitches[pitches.release_extension < 5.8]
    assert (long_ext.perceived_velo > long_ext.release_speed).all()
    assert (short_ext.perceived_velo < short_ext.release_speed).all()


def test_higher_slot_means_more_ride(built):
    _, _, curves = built
    ff = curves[2025]["Four-Seam"]
    assert ff["exp_ivb"][-1] > ff["exp_ivb"][0]  # over-the-top rides more
    assert ff["exp_hb"][-1] < ff["exp_hb"][0]    # lower slots run more


def test_planted_outliers_rank_high(built):
    _, a, _ = built
    q = a[a.n >= 50].dropna(subset=["slot_outlier"])
    pct = q.slot_outlier.rank(pct=True)
    for pid, (pt, *_r) in PLANTED.items():
        row = (q.pitcher == pid) & (q.pitch_type == pt)
        assert pct[row].iloc[0] > 0.9, f"{pid} {pt} not flagged"


def test_shrinkage_pulls_small_samples(built):
    _, a, _ = built
    small = a[a.n < 20].dropna(subset=["ivb_oe"])
    raw = (small.ivb - small.exp_ivb).abs()
    assert (small.ivb_oe.abs() <= raw + 1e-9).all()


def test_tilt():
    assert list(spin_axis_to_tilt(pd.Series([180.0, 90.0, 210.0]))) == ["12:00", "9:00", "1:00"]


def test_api(tmp_path, monkeypatch):
    import pitchlab.config as cfg
    monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
    import pitchlab.build as b
    monkeypatch.setattr(b, "PROCESSED_DIR", tmp_path)
    b.build({2025: make_sample(2025, n_pitchers=200)}, "sample", log=lambda *a: None)
    import app.main as m
    monkeypatch.setattr(m, "PROCESSED_DIR", tmp_path)
    c = TestClient(m.app)
    assert c.get("/api/seasons").json() == [2025]
    lb = c.get("/api/leaderboard", params={"season": 2025, "family": "Four-Seam"}).json()
    assert lb and lb[0]["slot_outlier"] >= lb[-1]["slot_outlier"]
    assert c.get("/api/pitcher/900001", params={"season": 2025}).status_code == 200
    assert c.post("/api/refresh", params={"season": 2025}).status_code == 403
    assert c.get("/api/analysis").json()["validation"]
    top = c.get("/api/leaderboard", params={"season": 2025, "sort": "stuff_plus", "limit": 5}).json()
    assert top[0]["stuff_plus"] > 110


def test_vaa_and_separation(built):
    pitches, _, _ = built
    assert pitches["vaa"].between(-15, 0).mean() > 0.99
    fb = pitches[pitches.d_velo == 0]
    ch = pitches[pitches.pitch_type == "CH"]
    assert len(fb) and ch["d_velo"].mean() < -5


def test_stuff_learns_direction():
    """Four-seamers should grade better with more ride, changeups with less."""
    from pitchlab.metrics import add_pitch_moe
    from pitchlab.stuff import fit_stuff, quintile_analysis, stuff_plus_table
    p = add_features(make_sample(2025, n_pitchers=260, seed=11))
    a, curves = fit_movement_models(aggregate_arsenal(p))
    p = add_pitch_moe(p, curves)
    graded, report = fit_stuff(p, families=["Four-Seam", "Changeup"], log=lambda *a: None)
    a = a.merge(stuff_plus_table(graded), on=["game_year", "pitcher", "pitch_type"], how="left")
    q = quintile_analysis(a)
    assert q["Four-Seam"]["ivb_oe"]["corr_stuff"] > 0.3
    assert q["Changeup"]["ivb_oe"]["corr_stuff"] < -0.3
    assert report["Four-Seam"]["+ arm slot & slot-adjusted movement"]["same_rv"] > 0.3
    v2 = report["Four-Seam"]["v2: + batter hand, outcome models, SL/ST pooled"]
    assert v2["same_rv"] > 0.3 and v2["same_whiff"] > 0.3
    assert "vaa_adj" in q["Four-Seam"] and "d_ivb" in q["Changeup"]


def test_v2_runs_with_platoon_column():
    """v2 trains and predicts with the batter-hand column present."""
    from pitchlab.stuff import event_values, _v2_oof, V2_FEATURES, make_target
    from pitchlab.metrics import add_pitch_moe
    p = add_features(make_sample(2025, n_pitchers=150, seed=5))
    a, curves = fit_movement_models(aggregate_arsenal(p))
    p = add_pitch_moe(p, curves)
    p = p[p.family == "Slider"].copy()
    from pitchlab.stuff import add_vaa_adj
    add_vaa_adj(p)
    p["target"] = make_target(p)
    p["same_side"] = (p["stand"] == p["p_throws"]).astype(float)
    p["is_sweeper"] = 0.0
    ev = event_values(p)
    rv1, _ = _v2_oof(p, ev, folds=3)
    p2 = p.copy()
    p2["same_side"] = 1.0  # every hitter same side
    # same rows, so only the platoon column differs; blended grades must match
    import numpy as np
    X = p[V2_FEATURES].to_numpy(float); X2 = p2[V2_FEATURES].to_numpy(float)
    assert not np.allclose(X, X2)
    assert np.isfinite(rv1).all()


def test_writeup_runs(tmp_path, monkeypatch):
    import pitchlab.build as b
    import pitchlab.writeup as w
    proc = tmp_path / "processed"
    proc.mkdir()
    monkeypatch.setattr(b, "PROCESSED_DIR", proc)
    monkeypatch.setattr(w, "PROCESSED_DIR", proc)
    monkeypatch.setattr(w, "OUT", tmp_path / "writeup")
    monkeypatch.setattr(w, "REPS", 20)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(w, "RAW_DIR", raw_dir)
    raws = {y: make_sample(y, n_pitchers=160) for y in (2025, 2026)}
    for y, r in raws.items():
        r.to_parquet(raw_dir / f"statcast_{y}.parquet", index=False)
    b.build(raws, "sample", log=lambda *a: None)
    w.main()
    md = (tmp_path / "writeup" / "results.md").read_text()
    assert "## Four-Seam" in md and "SOLID" in md and "Chase rate" in md and "Ground-ball rate" in md
    assert (tmp_path / "writeup" / "fourseam_ride_oe.png").exists()
