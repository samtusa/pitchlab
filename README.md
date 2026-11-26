# PitchLab

Pitch biomechanics and movement analytics built on public Statcast data. This is the
first module of the rebuilt baseball analytics site. Fantasy projections come next.

**What it does**
- **Slot outliers:** for every pitch type, models the movement an average pitch gets
  from a given arm angle and velocity, then finds pitchers whose movement is unusual
  for their slot.
- **Perceived velocity:** adjusts release speed for extension, so a 95 mph fastball
  released from 7.5 ft plays faster than the radar gun reading.
- **Stuff+:** our own pitch grade from physical traits only (no location), checked
  on pitchers the model never saw.
- **Does it matter?:** tests whether slot-adjusted movement actually predicts results,
  separately for each pitch type.
- **Pitcher pages:** arm angle, extension, release point, spin tilt, approach angle,
  separation from the fastball, Stuff+ and results, and actual vs expected movement
  for every pitch.

## Run it on your computer

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pitchlab.build 2025 2026       # download both seasons (~20 min each), train Stuff+, build
python -m uvicorn app.main:app --reload  # open http://localhost:8000
```

Always list every season you want in one command: Stuff+ is trained on all of them
together, and year-to-year validation needs at least two. Re-running only downloads
dates that aren't stored yet, and data saved by an older version without outcome
columns is downloaded again automatically. To try it with no network, run
`python -m pitchlab.build 2025 2026 --sample`. That builds a synthetic dataset and the site
shows a banner saying so. Delete `data/` before building real data. Run the tests
with `pytest`.

## Deploy (Render)

1. Push this folder to a GitHub repo.
2. On render.com, choose **New → Blueprint** and pick the repo. `render.yaml` sets
   up a Docker web service with a 2 GB persistent disk.
3. On first boot it builds the 2025 and 2026 seasons in the background. The site loads right
   away and the data appears when the build finishes.
4. With `PITCHLAB_AUTO_REFRESH=1` it pulls new games once a day from March to October.
   To refresh by hand, call
   `curl -X POST "https://<your-app>/api/refresh?season=2026" -H "X-Refresh-Token: <token>"`.
   The token is in the Render dashboard under Environment.

Any host that runs a Docker container and has a persistent volume works the same way
(Railway, Fly.io).

## How the metrics work

**Conventions.** Movement is *induced* break (gravity removed) in inches. Horizontal
values are mirrored for lefties so **positive means arm side** (run) and **negative
means glove side** (sweep, cut). Arm angle comes from Savant: 0° is sidearm, 90° is
straight over the top.

**Movement over expected (`pitchlab/metrics.py`).** The model is fit separately for
each pitch family (four-seam, sinker, cutter, slider, sweeper, curveball, changeup,
splitter):

    expected IVB, expected HB  ~  arm angle + arm angle² + release speed

The fit uses weighted least squares across all pitchers with 50 or more of that pitch.
Weights are pitch counts, capped at 600 so one workhorse can't dominate. Then:

- `ivb_oe` / `hb_oe`: actual minus expected, multiplied by n / (n + 75). The
  multiplier shrinks small samples toward zero, so 20 pitches can't top the board.
- `slot_outlier`: Mahalanobis distance of (ivb_oe, hb_oe) from the league's residual
  spread. 2 is rare and 3 is extreme.

Velocity is in the model because harder pitches have less time to move. The score
answers "unusual for this slot *and* this velocity."

**Perceived velocity (`pitchlab/features.py`).**

    perceived = release_speed × (60.5 − league_median_extension) / (60.5 − extension)

This matches reaction time from release. Savant's own `effective_speed` is shown next
to it as a check.

**Vertical approach angle.** Computed from Statcast's trajectory fit at the front of
the plate. For Stuff+ it is adjusted for pitch height (`vaa_adj`), because high pitches
are naturally flatter.

**Fastball separation.** For each pitch: velocity, IVB and HB minus the pitcher's
primary fastball. This matters most for changeups and splitters, which are judged
against his own fastball rather than the league.

**Stuff+ (`pitchlab/stuff.py`).** Version 1 is a gradient-boosted model per pitch
family that predicts run value from the pitcher's side. Balls in play are de-noised: each one is replaced
by the league-average value for contact of that quality (xwOBA). Location is not an
input. Cross-validation is grouped by pitcher, so every grade comes from a model that
never saw that pitcher in any season. 100 is average and 10 points is one standard
deviation.

Version 2, which the site shows, makes four changes. It adds the batter's handedness,
then blends each grade at the league platoon mix so a pitcher isn't credited for his
matchups. It trains sliders and sweepers together with a sweeper flag. It replaces the
single run-value target with separate models for swing, whiff per swing, ball in play
per contact, called strike per take, and contact quality, combined with league run
values for each outcome. And it uses stronger regularization. v1's three variants stay
in the comparison table so the two versions can be compared directly.

**What matters?** Three tests, all on the "What matters?" page:
1. *Quintiles:* each pitch family is split into five groups by IVB+ and by HB+, and
   the page shows actual run value, whiff rate and Stuff+ for each group. Direction is
   measured, not assumed.
2. *Ablation:* Stuff+ is trained three ways (movement only; + release point; + arm
   slot and slot-adjusted movement) and the three are compared on same-season and
   next-season results, alongside v2. Last season's actual results are a baseline. A
   good stuff model should predict next season better than last season's results do.
3. *Separation and approach angle:* the same quintile split by gap from the pitcher's
   own fastball (velocity, IVB, HB) for offspeed and breaking pitches, and by
   height-adjusted approach angle for fastballs.

## Write-up robustness checks

After a build, run `python -m pitchlab.writeup`. It needs no rebuild: chase, batted-ball and
hard-hit rates are computed straight from the raw pitch files. It writes
`data/writeup/results.md`, `results.json` and charts.

Outcomes: whiff rate, chase rate, ground-ball rate, pop-up rate and hard-hit rate.
xwOBA on contact and run value are in an appendix.

Traits are oriented so that bigger means "more of it": ride over slot, sweep or run over
slot, flatness of approach angle, and speed or drop gap from the fastball.

Each result is reported as top-20% vs bottom-20% averages, plus the effect per unit of the
trait with controls held fixed. Both carry 95% intervals from a bootstrap that resamples
pitchers, the per-unit effect is shown for each season separately, and every result gets a
verdict: SOLID, SUGGESTIVE or NOISE. The file also includes example pitchers and
model-validation correlations with confidence intervals.

## Layout

    pitchlab/config.py    settings, pitch families, kept columns
    pitchlab/ingest.py    Savant CSV downloader (chunked, incremental)
    pitchlab/features.py  mirroring, arm angle, perceived velo, spin tilt
    pitchlab/metrics.py   arsenal aggregation, slot model, pitcher summary
    pitchlab/stuff.py     Stuff+ model, validation, quintile analysis
    pitchlab/writeup.py   robustness checks and charts for write-ups
    pitchlab/build.py     CLI: raw -> processed tables in data/processed
    pitchlab/sample.py    synthetic data for tests and demos
    app/main.py           FastAPI JSON API + static site
    app/static/           frontend (plain JS + SVG, works on phones)

## Next steps
- Spin axis vs movement direction (seam-shifted wake detection)
- Release-point consistency and tunneling between pitches
- Compare Stuff+ with FanGraphs' Stuff+ for pitchers you look up
- Rebuild the fantasy projection pipeline on the same data layer, with the RBI fix
- Video module
