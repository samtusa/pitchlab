/* PitchLab frontend: plain JS + SVG, no build step. */
const FAMILIES = ["Four-Seam", "Sinker", "Cutter", "Slider", "Sweeper", "Curveball", "Changeup", "Splitter"];
const state = { season: null, family: "Four-Seam", hand: "", metric: "ivb", veloSort: "perceived_velo", meta: null };
const $ = (s, el = document) => el.querySelector(s);
const view = $("#view");
const tip = $("#tooltip");

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
const f1 = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d) + "%");
const signed = (v, d = 1) => (v == null ? "–" : (v > 0 ? "+" : "") + Number(v).toFixed(d));
const cls = (v, t = 0.5) => (v == null ? "" : v > t ? "pos" : v < -t ? "neg" : "");
const lastName = (n) => (n || "").split(",")[0];
const displayName = (n) => { const [l, f] = (n || "").split(", "); return f ? `${f} ${l}` : n; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function tilt(axis) {
  if (axis == null) return "–";
  let mins = Math.round((((axis / 30 + 6) % 12) * 60) / 15) * 15;
  let h = Math.floor(mins / 60) % 12 || 12, m = mins % 60;
  return `${h}:${String(m).padStart(2, "0")}`;
}

/* ---------- tooltip ---------- */
function showTip(evt, html) {
  tip.innerHTML = html; tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > innerHeight - 8) y = evt.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => (tip.hidden = true);
const row = (k, v) => `<div class="r"><span>${k}</span><span>${v}</span></div>`;

/* ---------- scatter chart ---------- */
function niceTicks(lo, hi, n = 6) {
  const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(6));
  return ticks;
}
function extent(vals, padFrac = 0.06, minSpan = 0) {
  const v = vals.filter((x) => x != null && isFinite(x));
  let lo = Math.min(...v), hi = Math.max(...v);
  if (hi - lo < minSpan) { const c = (hi + lo) / 2; lo = c - minSpan / 2; hi = c + minSpan / 2; }
  if (lo === hi) { lo -= 1; hi += 1; }
  const p = (hi - lo) * padFrac;
  return [lo - p, hi + p];
}

function scatter(el, o) {
  const W = Math.max(320, Math.min(o.width || 760, el.clientWidth || 760)), H = o.height || 420, m = { t: 12, r: 20, b: 44, l: 52 };
  const pts = o.points;
  const xs = pts.map((p) => p[o.x]).concat(o.extraX || []);
  const ys = pts.map((p) => p[o.y]).concat(o.extraY || []);
  const [x0, x1] = extent(xs, 0.08, o.minSpan || 0), [y0, y1] = extent(ys, 0.08, o.minSpan || 0);
  const sx = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const sy = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || "")}">`;
  s += `<g class="grid">`;
  for (const t of niceTicks(y0, y1)) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${m.l - 8}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
  for (const t of niceTicks(x0, x1, 8)) s += `<text x="${sx(t)}" y="${H - m.b + 16}" text-anchor="middle">${t}</text>`;
  s += `</g>`;
  if (y0 < 0 && y1 > 0) s += `<line class="zero" x1="${m.l}" x2="${W - m.r}" y1="${sy(0)}" y2="${sy(0)}"/>`;
  if (x0 < 0 && x1 > 0) s += `<line class="zero" x1="${sx(0)}" x2="${sx(0)}" y1="${m.t}" y2="${H - m.b}"/>`;
  s += `<text class="axis-title" x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${o.xLabel}</text>`;
  s += `<text class="axis-title" transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${o.yLabel}</text>`;
  if (o.diag) {
    const a = Math.max(x0, y0), b = Math.min(x1, y1);
    s += `<line x1="${sx(a)}" y1="${sy(a)}" x2="${sx(b)}" y2="${sy(b)}" stroke="var(--expected)" stroke-width="2" stroke-dasharray="5 4"/>`;
  }
  if (o.curve) {
    const d = o.curve.x.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(o.curve.y[i]).toFixed(1)}`).join("");
    s += `<path d="${d}" fill="none" stroke="var(--expected)" stroke-width="2"/>`;
  }
  if (o.pairs) { // connector from expected to actual
    for (const p of pts) if (p[o.pairs.x] != null)
      s += `<line x1="${sx(p[o.pairs.x])}" y1="${sy(p[o.pairs.y])}" x2="${sx(p[o.x])}" y2="${sy(p[o.y])}" stroke="var(--expected)" stroke-width="1.5"/>`;
    for (const p of pts) if (p[o.pairs.x] != null)
      s += `<circle cx="${sx(p[o.pairs.x])}" cy="${sy(p[o.pairs.y])}" r="5" fill="var(--surface-1)" stroke="var(--expected)" stroke-width="2"/>`;
  }
  const ordered = [...pts].sort((a, b) => (o.emph?.(a) ? 1 : 0) - (o.emph?.(b) ? 1 : 0));
  ordered.forEach((p) => {
    if (p[o.x] == null || p[o.y] == null) return;
    const hot = o.emph?.(p);
    const r = o.r ? o.r(p) : hot ? 5.5 : 4;
    const fill = hot ? "var(--series-2)" : "var(--series-1)";
    const op = hot || !o.emph ? 1 : 0.5;
    s += `<circle class="pt" data-i="${pts.indexOf(p)}" cx="${sx(p[o.x])}" cy="${sy(p[o.y])}" r="${r}" fill="${fill}" fill-opacity="${op}" stroke="var(--surface-1)" stroke-width="${hot ? 2 : 1}"/>`;
  });
  if (o.label) {
    // place labels at the first free spot around the point; skip if none is free
    const boxes = [];
    const hit = (b) => boxes.some((q) => b.x < q.x + q.w && q.x < b.x + b.w && b.y < q.y + q.h && q.y < b.y + b.h);
    pts.forEach((p) => {
      const t = o.label(p);
      if (!t || p[o.x] == null) return;
      const px = sx(p[o.x]), py = sy(p[o.y]), w = t.length * 6.6 + 2, h = 13;
      const spots = [[8, -7, "start"], [8, 15, "start"], [-8, -7, "end"], [-8, 15, "end"], [0, -12, "middle"]];
      for (const [dx, dy, anchor] of spots) {
        const x = anchor === "start" ? px + dx : anchor === "end" ? px + dx - w : px - w / 2;
        const b = { x, y: py + dy - 11, w, h };
        if (b.x < m.l || b.x + w > W || b.y < 0 || hit(b)) continue;
        boxes.push(b);
        s += `<text class="label" x="${px + dx}" y="${py + dy}" text-anchor="${anchor}">${esc(t)}</text>`;
        break;
      }
    });
  }
  s += `</svg>`;
  el.innerHTML = s;
  el.querySelectorAll(".pt").forEach((c) => {
    const p = pts[+c.dataset.i];
    c.style.cursor = o.onClick ? "pointer" : "default";
    c.addEventListener("mousemove", (e) => showTip(e, o.tooltip(p)));
    c.addEventListener("mouseleave", hideTip);
    if (o.onClick) c.addEventListener("click", () => { hideTip(); o.onClick(p); });
  });
}

/* ---------- table ---------- */
function table(el, rows, cols) {
  let h = `<div class="scroll"><table><thead><tr>${cols.map((c) => `<th class="${c.l ? "l" : ""}" title="${esc(c.title || "")}">${c.h}</th>`).join("")}</tr></thead><tbody>`;
  for (const r of rows) h += `<tr>${cols.map((c) => `<td class="${c.l ? "l" : ""} ${c.cls ? c.cls(r) : ""}">${c.f(r)}</td>`).join("")}</tr>`;
  el.innerHTML = h + `</tbody></table></div>`;
}
const pitcherLink = (r) => `<a href="#/pitcher/${r.pitcher}">${esc(displayName(r.player_name))}</a>`;

function seg(id, options, current, onPick) {
  const wrap = document.createElement("div");
  wrap.className = "seg"; wrap.id = id;
  options.forEach(([val, text]) => {
    const b = document.createElement("button");
    b.textContent = text; b.className = val === current ? "on" : "";
    b.onclick = () => onPick(val);
    wrap.append(b);
  });
  return wrap;
}

/* ---------- views ---------- */
async function outliersView() {
  const fam = state.family, isIvb = state.metric === "ivb";
  view.innerHTML = `
    <h1>Movement over slot expectation</h1>
    <p class="sub">For each pitch type, the gray line is the movement an average MLB pitch of that type gets from a given arm angle (at league-average velocity). Points far from the line move unusually for the pitcher's slot. Horizontal break is shown from the pitcher's arm side, so lefties and righties compare directly.</p>
    <div class="filters" id="filters"></div>
    <div class="card"><div class="legend" id="legend"></div><div id="chart"></div>
      <p class="note" id="model-note"></p></div>
    <div class="card"><h2>Biggest slot outliers: ${esc(fam)}</h2><div id="tbl"></div>
      <p class="note">IVB+ and HB+ are actual minus expected movement in inches, shrunk toward zero for small samples. Outlier score is the 2D distance from expected in standard deviations: 2+ is rare, 3+ is extreme.</p></div>`;
  const fl = $("#filters");
  fl.append(seg("fam", FAMILIES.map((f) => [f, f]), fam, (v) => { state.family = v; outliersView(); }));
  fl.append(seg("hand", [["", "All"], ["R", "RHP"], ["L", "LHP"]], state.hand, (v) => { state.hand = v; outliersView(); }));
  fl.append(seg("metric", [["ivb", "Vertical (IVB)"], ["hb", "Horizontal (HB)"]], state.metric, (v) => { state.metric = v; outliersView(); }));

  const [sc, lb] = await Promise.all([
    api(`/api/scatter?season=${state.season}&family=${encodeURIComponent(fam)}`),
    api(`/api/leaderboard?season=${state.season}&family=${encodeURIComponent(fam)}&limit=40${state.hand ? "&hand=" + state.hand : ""}`),
  ]);
  let pts = sc.points;
  if (state.hand) pts = pts.filter((p) => p.p_throws === state.hand);
  const top = new Set(lb.slice(0, 10).map((r) => r.pitcher + r.pitch_type));
  const lab = new Set(lb.slice(0, 5).map((r) => r.pitcher + r.pitch_type));
  $("#legend").innerHTML = `<span><i class="sw" style="background:var(--series-1)"></i>${esc(fam)}s (${pts.length})</span>
    <span><i class="sw" style="background:var(--series-2)"></i>Top 10 slot outliers</span>
    <span><i class="sw line" style="background:var(--expected)"></i>Expected for arm angle</span>`;
  if (!pts.length) { $("#chart").innerHTML = `<p class="sub">No qualifying pitchers.</p>`; }
  else scatter($("#chart"), {
    points: pts, x: "arm_angle", y: isIvb ? "ivb" : "hb",
    xLabel: "Arm angle (degrees; higher is more over-the-top)",
    yLabel: isIvb ? "Induced vertical break (in)" : "Horizontal break, arm side + (in)",
    aria: `${fam} arm angle versus ${isIvb ? "vertical" : "horizontal"} movement`,
    curve: sc.curve && { x: sc.curve.arm_angle, y: isIvb ? sc.curve.exp_ivb : sc.curve.exp_hb },
    extraY: sc.curve ? (isIvb ? sc.curve.exp_ivb : sc.curve.exp_hb) : [],
    emph: (p) => top.has(p.pitcher + p.pitch_type),
    label: (p) => (lab.has(p.pitcher + p.pitch_type) ? lastName(p.player_name) : null),
    onClick: (p) => (location.hash = `#/pitcher/${p.pitcher}`),
    tooltip: (p) => `<b>${esc(displayName(p.player_name))} · ${p.pitch_type}</b>` +
      row("Arm angle", f1(p.arm_angle) + "°") + row("Velo", f1(p.velo) + " mph") +
      row("IVB / expected", `${f1(p.ivb)} / ${f1(p.exp_ivb)}`) + row("HB / expected", `${f1(p.hb)} / ${f1(p.exp_hb)}`) +
      row("Outlier score", f1(p.slot_outlier, 2)) + row("Pitches", p.n),
  });
  if (sc.curve) $("#model-note").textContent =
    `Model fit on ${sc.curve.n_pitchers} pitchers at ${sc.curve.league_velo} mph league velo. Arm angle and velo explain ${Math.round(sc.curve.r2_ivb * 100)}% of the variation in IVB and ${Math.round(sc.curve.r2_hb * 100)}% in HB; typical miss is ±${sc.curve.sd_ivb} in vertical, ±${sc.curve.sd_hb} in horizontal.`;
  table($("#tbl"), lb, [
    { h: "Pitcher", l: 1, f: pitcherLink },
    { h: "T", f: (r) => r.p_throws }, { h: "Pitch", f: (r) => r.pitch_type }, { h: "N", f: (r) => r.n },
    { h: "Arm°", f: (r) => f1(r.arm_angle) }, { h: "Velo", f: (r) => f1(r.velo) },
    { h: "IVB", f: (r) => f1(r.ivb) }, { h: "xIVB", f: (r) => f1(r.exp_ivb) },
    { h: "IVB+", f: (r) => signed(r.ivb_oe), cls: (r) => cls(r.ivb_oe, 1.5) },
    { h: "HB", f: (r) => f1(r.hb) }, { h: "xHB", f: (r) => f1(r.exp_hb) },
    { h: "HB+", f: (r) => signed(r.hb_oe), cls: (r) => cls(r.hb_oe, 1.5) },
    { h: "Score", f: (r) => f1(r.slot_outlier, 2) },
  ]);
}

async function veloView() {
  view.innerHTML = `
    <h1>Perceived velocity</h1>
    <p class="sub">A pitcher with long extension lets go of the ball closer to the plate, so the hitter gets less time to react. Perceived velocity is the release speed that would give a hitter the same reaction time from league-average extension (${f1(state.meta?.league_extension_ft, 2)} ft). Points above the dashed line play faster than the radar gun says.</p>
    <div class="filters" id="filters"></div>
    <div class="card"><div class="legend"><span><i class="sw" style="background:var(--series-1)"></i>Primary fastballs (four-seam or sinker)</span>
      <span><i class="sw line" style="background:var(--expected)"></i>Perceived = actual</span></div><div id="chart"></div></div>
    <div class="card"><h2>Leaderboard</h2><div id="tbl"></div></div>`;
  $("#filters").append(seg("vs", [["perceived_velo", "Highest perceived velo"], ["velo_gain", "Biggest extension boost"], ["extension", "Longest extension"]],
    state.veloSort, (v) => { state.veloSort = v; veloView(); }));
  const [ff, si] = await Promise.all(["Four-Seam", "Sinker"].map((f) =>
    api(`/api/leaderboard?season=${state.season}&family=${f}&sort=${state.veloSort}&limit=500&min_pitches=100`)));
  // keep each pitcher's most-used fastball
  const best = new Map();
  for (const r of [...ff, ...si]) if (!best.has(r.pitcher) || best.get(r.pitcher).n < r.n) best.set(r.pitcher, r);
  const rows = [...best.values()].sort((a, b) => b[state.veloSort] - a[state.veloSort]);
  scatter($("#chart"), {
    points: rows, x: "velo", y: "perceived_velo", diag: true, height: 380,
    xLabel: "Release speed (mph)", yLabel: "Perceived velocity (mph)",
    aria: "Release speed versus perceived velocity",
    label: (p) => (rows.indexOf(p) < 4 ? lastName(p.player_name) : null),
    onClick: (p) => (location.hash = `#/pitcher/${p.pitcher}`),
    tooltip: (p) => `<b>${esc(displayName(p.player_name))} · ${p.pitch_type}</b>` + row("Release speed", f1(p.velo) + " mph") +
      row("Extension", f1(p.extension, 2) + " ft") + row("Perceived", f1(p.perceived_velo) + " mph") +
      row("Boost", signed(p.velo_gain) + " mph") + row("Savant effective", f1(p.effective_speed) + " mph"),
  });
  table($("#tbl"), rows.slice(0, 50), [
    { h: "Pitcher", l: 1, f: pitcherLink }, { h: "T", f: (r) => r.p_throws }, { h: "Pitch", f: (r) => r.pitch_type },
    { h: "Velo", f: (r) => f1(r.velo) }, { h: "Ext (ft)", f: (r) => f1(r.extension, 2) },
    { h: "Perceived", f: (r) => f1(r.perceived_velo) },
    { h: "Boost", f: (r) => signed(r.velo_gain), cls: (r) => cls(r.velo_gain, 0.5) },
    { h: "Savant eff.", f: (r) => f1(r.effective_speed), title: "Baseball Savant's own effective velocity, for comparison" },
  ]);
}

async function pitcherView(pid) {
  const d = await api(`/api/pitcher/${pid}?season=${state.season}`);
  const s = d.summary, ars = d.arsenal;
  view.innerHTML = `
    <a href="#/">← Back</a>
    <h1>${esc(displayName(s.player_name))} <span class="sub">${s.p_throws}HP · ${state.season}</span></h1>
    <div class="stats">
      <div class="stat"><div class="v">${f1(s.arm_angle)}°</div><div class="k">Arm angle</div></div>
      <div class="stat"><div class="v">${f1(s.extension, 2)} ft</div><div class="k">Extension</div></div>
      <div class="stat"><div class="v">${f1(s.fb_velo)}</div><div class="k">${s.fb_type || "FB"} velo (mph)</div></div>
      <div class="stat"><div class="v">${f1(s.fb_perceived)}</div><div class="k">Perceived velo (mph)</div></div>
      <div class="stat"><div class="v">${s.stuff_plus == null ? "–" : Math.round(s.stuff_plus)}</div><div class="k">Stuff+ (all pitches)</div></div>
      <div class="stat"><div class="v">${s.pitches}</div><div class="k">Pitches</div></div>
    </div>
    <div class="grid2">
      <div class="card"><h2>Movement vs. slot expectation</h2>
        <div class="legend"><span><i class="sw" style="background:var(--series-1)"></i>Actual</span><span><i class="sw ring"></i>Expected for his arm angle</span></div>
        <div id="mv"></div>
        <p class="note">Arm side is to the right. The gray line shows how far each pitch moves from what his slot predicts.</p></div>
      <div class="card"><h2>Release point</h2>
        <div class="legend"><span><i class="sw" style="background:var(--series-1)"></i>Release by pitch (catcher's view, mirrored to arm side)</span></div>
        <div id="rel"></div>
        <p class="note">Tight clusters mean every pitch comes out of the same window, which makes them harder to tell apart.</p></div>
    </div>
    <div class="card"><h2>Arsenal</h2><div id="tbl"></div></div>`;
  const tipFn = (p) => `<b>${p.pitch_type} · ${p.family}</b>` + row("Usage", f1(p.usage * 100, 0) + "%") +
    row("IVB / expected", `${f1(p.ivb)} / ${f1(p.exp_ivb)}`) + row("HB / expected", `${f1(p.hb)} / ${f1(p.exp_hb)}`) +
    row("Outlier score", f1(p.slot_outlier, 2));
  scatter($("#mv"), {
    points: ars, x: "hb", y: "ivb", pairs: { x: "exp_hb", y: "exp_ivb" }, width: 480, height: 400, minSpan: 10,
    extraX: ars.map((p) => p.exp_hb), extraY: ars.map((p) => p.exp_ivb),
    xLabel: "Horizontal break, arm side + (in)", yLabel: "Induced vertical break (in)",
    aria: "Pitch movement versus expected", r: () => 7, label: (p) => p.pitch_type, tooltip: tipFn,
  });
  scatter($("#rel"), {
    points: ars, x: "release_x_arm", y: "release_z", width: 480, height: 400, minSpan: 1, r: () => 7,
    xLabel: "Release side, arm side + (ft)", yLabel: "Release height (ft)", aria: "Release point by pitch",
    label: (p) => p.pitch_type,
    tooltip: (p) => `<b>${p.pitch_type}</b>` + row("Side", f1(p.release_x_arm, 2) + " ft") + row("Height", f1(p.release_z, 2) + " ft") + row("Extension", f1(p.extension, 2) + " ft"),
  });
  table($("#tbl"), ars, [
    { h: "Pitch", l: 1, f: (r) => `${r.pitch_type} <span class="sub">${r.family}</span>` },
    { h: "Usage", f: (r) => f1(r.usage * 100, 0) + "%" }, { h: "N", f: (r) => r.n },
    { h: "Stuff+", f: (r) => (r.stuff_plus == null ? "–" : Math.round(r.stuff_plus)), cls: (r) => cls(r.stuff_plus == null ? null : r.stuff_plus - 100, 5) },
    { h: "RV/100", f: (r) => signed(r.rv100, 2), title: "Actual run value per 100 pitches, pitcher's view (+ is good)" },
    { h: "Whiff%", f: (r) => pct(r.whiff_pct) },
    { h: "Velo", f: (r) => f1(r.velo) }, { h: "Perceived", f: (r) => f1(r.perceived_velo) },
    { h: "VAA", f: (r) => f1(r.vaa), title: "Vertical approach angle at the plate (degrees). Flatter (closer to 0) fastballs miss more bats up in the zone." },
    { h: "ΔVelo", f: (r) => (r.d_velo === 0 ? "FB" : signed(r.d_velo)), title: "Velocity difference from his primary fastball" },
    { h: "ΔIVB", f: (r) => (r.d_velo === 0 ? "FB" : signed(r.d_ivb)), title: "Vertical break difference from his primary fastball" },
    { h: "Spin", f: (r) => (r.spin_rate ? Math.round(r.spin_rate) : "–") },
    { h: "Tilt", f: (r) => tilt(r.spin_axis), title: "Spin axis as a clock face (12:00 = pure backspin)" },
    { h: "Arm°", f: (r) => f1(r.arm_angle) },
    { h: "IVB", f: (r) => f1(r.ivb) }, { h: "IVB+", f: (r) => signed(r.ivb_oe), cls: (r) => cls(r.ivb_oe, 1.5) },
    { h: "HB", f: (r) => f1(r.hb) }, { h: "HB+", f: (r) => signed(r.hb_oe), cls: (r) => cls(r.hb_oe, 1.5) },
    { h: "Score", f: (r) => f1(r.slot_outlier, 2) },
  ]);
}

/* ---------- bar chart (single series, values can be negative) ---------- */
function bars(el, o) {
  const W = Math.max(300, Math.min(o.width || 480, el.clientWidth || 480)), H = o.height || 260;
  const m = { t: 14, r: 12, b: 48, l: 50 };
  const vals = o.data.map((d) => d.value);
  let [y0, y1] = extent(vals.concat([0]), 0.12);
  if (y0 > 0) y0 = 0; if (y1 < 0) y1 = 0;
  const sy = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const bw = (W - m.l - m.r) / o.data.length;
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || "")}"><g class="grid">`;
  for (const t of niceTicks(y0, y1, 5)) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${m.l - 8}" y="${sy(t) + 4}" text-anchor="end">${o.fmt ? o.fmt(t) : t}</text>`;
  s += `</g><line class="zero" x1="${m.l}" x2="${W - m.r}" y1="${sy(0)}" y2="${sy(0)}"/>`;
  o.data.forEach((d, i) => {
    const x = m.l + i * bw + 3, w = bw - 6, ya = sy(Math.max(0, d.value)), yb = sy(Math.min(0, d.value));
    s += `<rect class="bar" data-i="${i}" x="${x}" y="${ya}" width="${w}" height="${Math.max(1, yb - ya)}" rx="3" fill="var(--series-1)"/>`;
    s += `<text x="${x + w / 2}" y="${H - m.b + 16}" text-anchor="middle">${esc(d.label)}</text>`;
    if (d.sub && bw >= 70) s += `<text x="${x + w / 2}" y="${H - m.b + 30}" text-anchor="middle">${esc(d.sub)}</text>`;
  });
  s += `<text class="axis-title" x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">${o.xLabel}</text></svg>`;
  el.innerHTML = s;
  el.querySelectorAll(".bar").forEach((b) => {
    const d = o.data[+b.dataset.i];
    b.addEventListener("mousemove", (e) => showTip(e, o.tooltip(d)));
    b.addEventListener("mouseleave", hideTip);
  });
}

async function stuffView() {
  const fam = state.stuffFamily || "";
  view.innerHTML = `
    <h1>Stuff+</h1>
    <p class="sub">How good a pitch is based only on its physical traits: velocity, movement, spin, extension, release point, arm slot, approach angle, and separation from the pitcher's fastball. Location is left out. 100 is a league-average pitch, and every 10 points is one standard deviation. Each grade comes from a model that never saw that pitcher, so it's a fair test of the pitch, not a memory of its results.</p>
    <div class="filters" id="filters"></div>
    <div class="card"><div class="legend"><span><i class="sw" style="background:var(--series-1)"></i>Pitch types with 100+ pitches</span></div><div id="chart"></div>
      <p class="note">Run value per 100 pitches is from the pitcher's side, so higher is better. One season of results is noisy: a pitch can grade well and still get hit, or the reverse.</p></div>
    <div class="card"><h2>Top Stuff+ ${fam ? "· " + esc(fam) : ""}</h2><div id="tbl"></div></div>`;
  $("#filters").append(seg("sfam", [["", "All pitches"], ...FAMILIES.map((f) => [f, f])], fam, (v) => { state.stuffFamily = v; stuffView(); }));
  const rows = await api(`/api/leaderboard?season=${state.season}&sort=stuff_plus&limit=3000&min_pitches=100${fam ? "&family=" + encodeURIComponent(fam) : ""}`);
  if (!rows.length) { $("#chart").innerHTML = `<p class="sub">No Stuff+ grades yet. Rebuild with outcome data: <code>python -m pitchlab.build 2025 2026</code>.</p>`; return; }
  scatter($("#chart"), {
    points: rows, x: "stuff_plus", y: "rv100", height: 380,
    xLabel: "Stuff+ (model grade)", yLabel: "Actual run value per 100 pitches",
    aria: "Stuff+ versus actual run value",
    label: (p) => (rows.indexOf(p) < 4 ? `${lastName(p.player_name)} ${p.pitch_type}` : null),
    onClick: (p) => (location.hash = `#/pitcher/${p.pitcher}`),
    tooltip: (p) => `<b>${esc(displayName(p.player_name))} · ${p.pitch_type}</b>` + row("Stuff+", Math.round(p.stuff_plus)) +
      row("RV/100", signed(p.rv100, 2)) + row("Whiff%", pct(p.whiff_pct)) + row("IVB+ / HB+", `${signed(p.ivb_oe)} / ${signed(p.hb_oe)}`) + row("Pitches", p.n),
  });
  table($("#tbl"), rows.slice(0, 50), [
    { h: "Pitcher", l: 1, f: pitcherLink }, { h: "T", f: (r) => r.p_throws }, { h: "Pitch", f: (r) => r.pitch_type },
    { h: "N", f: (r) => r.n }, { h: "Stuff+", f: (r) => Math.round(r.stuff_plus) },
    { h: "RV/100", f: (r) => signed(r.rv100, 2), cls: (r) => cls(r.rv100, 0.75) },
    { h: "Whiff%", f: (r) => pct(r.whiff_pct) }, { h: "Velo", f: (r) => f1(r.velo) },
    { h: "IVB", f: (r) => f1(r.ivb) }, { h: "HB", f: (r) => f1(r.hb) },
    { h: "IVB+", f: (r) => signed(r.ivb_oe) }, { h: "HB+", f: (r) => signed(r.hb_oe) },
    { h: "VAA", f: (r) => f1(r.vaa) },
    { h: "ΔVelo", f: (r) => (r.d_velo === 0 ? "FB" : signed(r.d_velo)) },
  ]);
}

async function matterView() {
  const fam = state.family, metric = state.matterMetric || "rv100";
  const d = await api("/api/analysis");
  const V = d.validation?.[fam], Q = d.quintiles?.[fam];
  const seasons = d.seasons.join(" & ");
  view.innerHTML = `
    <h1>What actually matters?</h1>
    <p class="sub">Three tests on ${seasons} data. First, whether pitches with more movement than their slot predicts actually get better results, broken out by pitch type so direction can differ (a four-seamer may want more ride while a changeup wants less). Second, fastball separation (offspeed and breaking pitches) or approach angle (fastballs). Third, which version of the Stuff+ model best predicts results for pitchers it has never seen.</p>
    <div class="filters" id="filters"></div>
    <div class="card"><h2>Results by slot-adjusted movement: ${esc(fam)}</h2>
      <p class="note" style="margin:0 0 8px">Pitch types (100+ pitches) are split into five equal groups, from least to most movement over expected.</p>
      <div class="grid2"><div><h2 style="font-size:13px">Vertical (IVB+)</h2><div id="q-ivb"></div><p class="note" id="c-ivb"></p></div>
      <div><h2 style="font-size:13px">Horizontal, arm side + (HB+)</h2><div id="q-hb"></div><p class="note" id="c-hb"></p></div></div></div>
    <div class="card" id="extra-card" hidden><h2 id="extra-title"></h2>
      <p class="note" style="margin:0 0 8px" id="extra-sub"></p><div class="grid2" id="extra"></div></div>
    <div class="card"><h2>Which Stuff+ model predicts best? ${esc(fam)}</h2><div id="abl"></div><p class="note" id="abl-note"></p></div>`;
  const fl = $("#filters");
  fl.append(seg("fam", FAMILIES.map((f) => [f, f]), fam, (v) => { state.family = v; matterView(); }));
  fl.append(seg("mm", [["rv100", "Run value/100"], ["whiff_pct", "Whiff%"], ["stuff_plus", "Stuff+"]], metric, (v) => { state.matterMetric = v; matterView(); }));

  const fmt = metric === "whiff_pct" ? (v) => pct(v, 0) : metric === "rv100" ? (v) => signed(v, Math.abs(v * 10 - Math.round(v * 10)) > 1e-6 ? 2 : 1) : (v) => Math.round(100 + v);
  const corrKey = { rv100: "corr_rv100", whiff_pct: "corr_whiff", stuff_plus: "corr_stuff" }[metric];
  const mName = { rv100: "run value", whiff_pct: "whiff rate", stuff_plus: "Stuff+" }[metric];
  const EXTRA = {
    d_velo: ["Velo gap from his fastball", "mph (more negative = slower than the fastball)"],
    d_ivb: ["Vertical break gap from his fastball", "inches (more negative = drops more than the fastball)"],
    d_hb: ["Horizontal break gap from his fastball", "inches, arm side +"],
    vaa_adj: ["Approach angle vs. expected for pitch height", "degrees (positive = flatter than expected)"],
  };
  const extraCols = Q ? Object.keys(EXTRA).filter((c) => Q[c]) : [];
  if (extraCols.length) {
    $("#extra-card").hidden = false;
    const isFb = extraCols.includes("vaa_adj");
    $("#extra-title").textContent = isFb ? `Approach angle: ${fam}` : `Separation from the fastball: ${fam}`;
    $("#extra-sub").textContent = isFb
      ? "Approach angle depends on how high the pitch is thrown, so this compares each pitch with the angle expected at its height."
      : "Offspeed and breaking pitches are judged against the pitcher's own fastball, not the league. Gaps are this pitch minus his primary fastball.";
    $("#extra").innerHTML = extraCols.map((c) => `<div><h2 style="font-size:13px">${EXTRA[c][0]}</h2><div id="q-${c}"></div><p class="note" id="c-${c}"></p></div>`).join("");
  }
  for (const col of ["ivb_oe", "hb_oe", ...extraCols]) {
    const el = $(col === "ivb_oe" ? "#q-ivb" : col === "hb_oe" ? "#q-hb" : "#q-" + col);
    if (!Q) { el.innerHTML = `<p class="sub">Not enough data for this pitch type.</p>`; continue; }
    const qb = Q[col];
    bars(el, {
      data: qb.bins.map((b) => ({ ...b, value: metric === "stuff_plus" ? (b.stuff_plus ?? 100) - 100 : (b[metric] ?? 0), label: `Q${b.quintile}`, sub: `${signed(b.lo)} to ${signed(b.hi)}` })),
      xLabel: EXTRA[col] ? EXTRA[col][1] : "Movement over expected, inches", fmt, aria: `${mName} by ${col} quintile`,
      tooltip: (b) => `<b>Q${b.quintile}: ${signed(b.lo)} to ${signed(b.hi)}</b>` + row("RV/100", signed(b.rv100, 2)) +
        row("Whiff%", pct(b.whiff_pct)) + row("Stuff+", b.stuff_plus == null ? "–" : f1(b.stuff_plus, 0)) + row("Pitch types", b.count),
    });
    const r = qb[corrKey];
    $(col === "ivb_oe" ? "#c-ivb" : col === "hb_oe" ? "#c-hb" : "#c-" + col).textContent = r == null ? "" :
      `Correlation with ${mName}: ${signed(r, 2)}. ${Math.abs(r) < 0.1 ? "Essentially no relationship." : r > 0 ? "More movement in this direction goes with better results." : "Less movement in this direction goes with better results."}`;
  }

  if (!V) { $("#abl").innerHTML = `<p class="sub">No Stuff+ validation for this pitch type yet.</p>`; return; }
  const names = Object.keys(V), base = V[names[1]], full = V[names[2]], first = V[names[0]], v2 = V[names[3]];
  const cell = (v) => (v == null ? "–" : f1(v, 3));
  let h = `<div class="scroll"><table><thead><tr><th class="l">Model features</th><th title="Correlation with same-season run value">Same-season RV</th><th>Same-season whiff%</th><th title="This season's grade vs next season's results">Next-season RV</th><th>Next-season whiff%</th></tr></thead><tbody>`;
  for (const n of names) h += `<tr><td class="l">${n.startsWith("v2") ? "<b>" + esc(n) + "</b> (shown as Stuff+)" : "v1 " + esc(n)}</td><td>${cell(V[n].same_rv)}</td><td>${cell(V[n].same_whiff)}</td><td>${cell(V[n].next_rv)}</td><td>${cell(V[n].next_whiff)}</td></tr>`;
  if (first.next_rv_from_results != null)
    h += `<tr><td class="l"><i>Last season's actual results (baseline)</i></td><td>–</td><td>–</td><td>${cell(first.next_rv_from_results)}</td><td>${cell(first.next_whiff_from_results)}</td></tr>`;
  $("#abl").innerHTML = h + `</tbody></table></div>`;
  const key = full.next_rv != null ? "next_rv" : "same_rv", wkey = full.next_whiff != null ? "next_whiff" : "same_whiff";
  const which = key === "next_rv" ? "next-season" : "same-season";
  const gain = full[key] - base[key];
  let note = `Numbers are correlations (r) at the pitch-type level; higher means the grade tracks results better. ` +
    `In v1, adding arm slot and slot-adjusted movement changed ${which} run-value accuracy by ${signed(gain, 3)}` +
    (Math.abs(gain) < 0.01 ? " (within noise). " : gain > 0 ? " (slot-relative movement helped). " : " (it hurt). ");
  if (v2) note += `Version 2 vs. v1's best: ${signed(v2[key] - full[key], 3)} on run value, ${signed(v2[wkey] - full[wkey], 3)} on whiffs. v2's whiff columns use its own whiff-per-swing model. `;
  if (first.next_rv_from_results != null) note += `Model rows that beat the baseline row predict next year better than this year's results do.`;
  $("#abl-note").textContent = note;
}

/* ---------- router & boot ---------- */
async function route() {
  hideTip();
  const h = location.hash || "#/";
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active",
    a.dataset.nav === "outliers" ? h === "#/" : h.startsWith("#/" + a.dataset.nav)));
  try {
    const m = h.match(/^#\/pitcher\/(\d+)/);
    if (m) await pitcherView(m[1]);
    else if (h.startsWith("#/velo")) await veloView();
    else if (h.startsWith("#/stuff")) await stuffView();
    else if (h.startsWith("#/matter")) await matterView();
    else await outliersView();
  } catch (e) {
    view.innerHTML = `<p class="sub">Couldn't load data: ${esc(e.message)}</p>`;
  }
  window.scrollTo(0, 0);
}

async function boot() {
  const seasons = await api("/api/seasons").catch(() => []);
  if (!seasons.length) { view.innerHTML = `<p class="sub">No data built yet. Run <code>python -m pitchlab.build 2025</code>.</p>`; return; }
  const sel = $("#season");
  sel.innerHTML = seasons.map((y) => `<option>${y}</option>`).join("");
  state.season = seasons[0];
  const loadMeta = async () => {
    state.meta = await api(`/api/meta/${state.season}`);
    const b = $("#banner");
    b.hidden = state.meta.source !== "sample";
    b.textContent = "Showing synthetic sample data for testing. Names and numbers are not real.";
  };
  await loadMeta();
  sel.onchange = async () => { state.season = +sel.value; await loadMeta(); route(); };

  const input = $("#search"), list = $("#search-results");
  let timer;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 2) { list.hidden = true; return; }
    timer = setTimeout(async () => {
      const hits = await api(`/api/search?season=${state.season}&q=${encodeURIComponent(q)}`);
      list.innerHTML = hits.map((h) => `<li><a href="#/pitcher/${h.pitcher}">${esc(displayName(h.player_name))} <span class="sub">${h.p_throws}HP</span></a></li>`).join("") || `<li class="sub" style="padding:6px 8px">No match</li>`;
      list.hidden = false;
    }, 200);
  });
  list.addEventListener("click", () => { list.hidden = true; input.value = ""; });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) list.hidden = true; });

  addEventListener("hashchange", route);
  route();
}
boot();
