// Centered Ablation project page

// TODO: set to the arXiv abstract URL (e.g. https://arxiv.org/abs/XXXX.XXXXX) once announced
const ARXIV_URL = '';

// ---------------------------------------------------------------------------------------------
// Data, from the paper's appendix tables (mean over 6 calibration draws, s = 1, last layer).
//   asr: fluent-only attack success on HarmBench, HarmBench classifier (%)        [tab:asr]
//   rw : benign Alpaca answers rewritten (%)                                      [tab:faith]
//   col: collapse rate (%)                                                        [tab:asr]
//   gain: CA - DIM fluent-only ASR, paired prompt bootstrap (points)              [tab:asrdiff]
//   kept: benign answers preserved by CA relative to DIM (points)                 [tab:faith]
//   pilot: the 3 models the analysis was developed on; the other 5 were predicted in advance
// ---------------------------------------------------------------------------------------------
const MODELS = [
    { name: 'Qwen2.5-0.5B', omega: 0.022, eta: 0.679, pilot: false,
      asr: { none: 23.8, dim: 49.7, off: 49.7, ca: 49.7 }, rw: { dim: 59.2, off: 60.7, ca: 60.0 }, col: { dim: 3, off: 3, ca: 3 }, gain: 0.0, kept: -0.8 },
    { name: 'Qwen3-4B', omega: 0.034, eta: 0.963, pilot: false,
      asr: { none: 2.1, dim: 7.4, off: 7.6, ca: 7.2 }, rw: { dim: 30.2, off: 29.5, ca: 30.3 }, col: { dim: 0, off: 0, ca: 0 }, gain: -0.3, kept: -0.2 },
    { name: 'Qwen2.5-7B', omega: 0.041, eta: 0.888, pilot: true,
      asr: { none: 11.7, dim: 52.3, off: 53.9, ca: 50.1 }, rw: { dim: 35.2, off: 41.0, ca: 33.8 }, col: { dim: 0, off: 0, ca: 0 }, gain: -2.2, kept: 1.3 },
    { name: 'Qwen2.5-3B', omega: 0.125, eta: 0.921, pilot: false,
      asr: { none: 6.7, dim: 38.6, off: 53.3, ca: 41.0 }, rw: { dim: 67.5, off: 55.0, ca: 62.8 }, col: { dim: 0, off: 0, ca: 0 }, gain: 2.4, kept: 4.7 },
    { name: 'Gemma-2-2B', omega: 0.315, eta: 0.979, pilot: false,
      asr: { none: 2.1, dim: 29.3, off: 27.2, ca: 26.7 }, rw: { dim: 87.5, off: 69.7, ca: 71.0 }, col: { dim: 1, off: 0, ca: 0 }, gain: -2.6, kept: 16.5 },
    { name: 'Llama-3.1-8B', omega: 0.332, eta: 0.458, pilot: true,
      asr: { none: 9.2, dim: 33.6, off: 65.5, ca: 62.6 }, rw: { dim: 45.8, off: 72.8, ca: 33.5 }, col: { dim: 0, off: 0, ca: 0 }, gain: 29.0, kept: 12.3 },
    { name: 'Llama-3.2-3B', omega: 0.438, eta: 0.083, pilot: false,
      asr: { none: 6.7, dim: 10.3, off: 69.1, ca: 56.7 }, rw: { dim: 44.2, off: 71.7, ca: 25.7 }, col: { dim: 0, off: 0, ca: 0 }, gain: 46.3, kept: 18.5 },
    { name: 'Gemma-2-9B', omega: 0.471, eta: 0.979, pilot: true,
      asr: { none: 2.1, dim: 14.4, off: 13.9, ca: 14.0 }, rw: { dim: 73.2, off: 72.2, ca: 32.8 }, col: { dim: 50, off: 1, ca: 1 }, gain: -0.5, kept: 40.3 },
];
// judged harmful (%), Gemma-2-9B under DIM ablation, 120 degenerate + 120 fluent outputs [results/analysis/judge_artifact.json]
const JUDGES = { degen: { hb: 1.7, lg: 98.3 }, fluent: { hb: 29.2, lg: 32.5 } };

const C = { ca: '#2b6cb0', dim: '#9ca3af', dimInk: '#6b7280', off: '#d9893a', ink: '#1a2230', ink2: '#475569', ink3: '#64748b', line: '#e2e8f0', bad: '#b5473a', tint: '#f6f8fb' };

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs = {}, parent = null, text = null) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    // Computer Modern runs small and thin on screen, and charts render at ~0.9x in the 960px column: set chart text 20% larger
    if ((tag === 'text' || tag === 'tspan') && attrs['font-size']) e.setAttribute('font-size', (+attrs['font-size'] * 1.2).toFixed(1));
    if (text !== null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
}
const tip = () => document.getElementById('tooltip');
function showTip(evt, html) {
    const t = tip(); t.innerHTML = html; t.classList.add('show');
    const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    t.style.left = x + 'px'; t.style.top = y + 'px';
}
function hideTip() { tip().classList.remove('show'); }
const fmt = (v, d = 1) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d);

// seeded normal draws, so the toy picture is the same on every load
function rng(seed) { return () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; }; }
function normals(n, seed) {
    const r = rng(seed), out = [];
    for (let i = 0; i < n; i++) { const u = Math.max(r(), 1e-9), v = r(); out.push([Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v), Math.sqrt(-2 * Math.log(u)) * Math.sin(2 * Math.PI * v), r() * 2 - 1]); }
    return out;
}

// ---------------------------------------------------------------------------------------------
// 1. Geometry widget: toy 2D picture of the gate floor and the fix
// ---------------------------------------------------------------------------------------------
const Geo = {
    dir: 'dim', omega: 0.44, t: 0, target: 0,
    N: 36, R: 3.0, D: 1.7, SD: 0.3, ANG: 75 * Math.PI / 180,
    zB: normals(36, 7), zH: normals(36, 11),
    k: 66, ox: 104, oy: 292,
};
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const norm = a => Math.hypot(a[0], a[1]);
const unit = a => { const n = norm(a); return [a[0] / n, a[1] / n]; };
const sx = x => Geo.ox + Geo.k * x, sy = y => Geo.oy - Geo.k * y;

function geoState() {
    const m = [Math.cos(Geo.ANG), Math.sin(Geo.ANG)];       // common-mode axis
    const n = [Math.sin(Geo.ANG), -Math.cos(Geo.ANG)];      // orthogonal axis
    const w = Geo.omega, b = [Geo.R * m[0], Geo.R * m[1]];
    const d = [Geo.D * (-w * m[0] + Math.sqrt(1 - w * w) * n[0]), Geo.D * (-w * m[1] + Math.sqrt(1 - w * w) * n[1])];
    const hc = [b[0] + d[0], b[1] + d[1]];
    const B = Geo.zB.map(z => [b[0] + Geo.SD * z[0], b[1] + Geo.SD * z[1]]);
    const H = Geo.zH.map(z => [hc[0] + Geo.SD * z[0], hc[1] + Geo.SD * z[1]]);
    // the algorithm, on the toy samples
    const mean = P => [P.reduce((s, p) => s + p[0], 0) / P.length, P.reduce((s, p) => s + p[1], 0) / P.length];
    const mB = mean(B), mH = mean(H);
    const uDim = unit([mH[0] - mB[0], mH[1] - mB[1]]);
    const mu = unit(mean(B.map(unit)));
    const pr = dot(uDim, mu);
    const uCa = unit([uDim[0] - pr * mu[0], uDim[1] - pr * mu[1]]);
    const u = Geo.dir === 'ca' ? uCa : uDim;
    const gate = p => dot(p, u) / norm(p);
    const abl = p => { const c = dot(p, u); return [p[0] - c * u[0], p[1] - c * u[1]]; };
    return { mu, u, B, H, gB: B.map(gate), gH: H.map(gate), aB: B.map(abl), aH: H.map(abl), omegaMeasured: Math.abs(pr) };
}

function drawGeo() {
    const svg = document.getElementById('geo-svg'); if (!svg) return;
    svg.innerHTML = '';
    const S = geoState(), t = Geo.t, col = Geo.dir === 'ca' ? C.ca : C.dimInk;
    const defs = el('defs', {}, svg);
    for (const [id, c] of [['arr-u', col], ['arr-m', C.ink3]]) {
        const mk = el('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
        el('path', { d: 'M0,0 L10,5 L0,10 z', fill: c }, mk);
    }
    // faint grid
    for (let gx = -1; gx <= 4; gx++) el('line', { x1: sx(gx), y1: 8, x2: sx(gx), y2: 352, stroke: '#f1f5f9' }, svg);
    for (let gy = -1; gy <= 4; gy++) el('line', { x1: 8, y1: sy(gy), x2: 412, y2: sy(gy), stroke: '#f1f5f9' }, svg);

    // common-mode ray
    const L = 4.7;
    el('line', { x1: sx(0), y1: sy(0), x2: sx(L * S.mu[0]), y2: sy(L * S.mu[1]), stroke: C.ink3, 'stroke-width': 1.3, 'stroke-dasharray': '5 4' }, svg);
    el('text', { x: sx(4.2 * S.mu[0]) + 8, y: sy(4.2 * S.mu[1]) + 4, 'font-size': 11, fill: C.ink3 }, svg, 'common mode');

    // ablation hyperplane (states land on it at s = 1)
    const pv = [-S.u[1], S.u[0]], P = 6;
    el('line', { x1: sx(-P * pv[0]), y1: sy(-P * pv[1]), x2: sx(P * pv[0]), y2: sy(P * pv[1]), stroke: col, 'stroke-opacity': 0.12 + 0.35 * t, 'stroke-width': 1.5 }, svg);

    // displacement segments + points
    const lerp = (a, b) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    if (t > 0) {
        S.B.forEach((p, i) => { const q = lerp(p, S.aB[i]); el('line', { x1: sx(p[0]), y1: sy(p[1]), x2: sx(q[0]), y2: sy(q[1]), stroke: '#64748b', 'stroke-opacity': 0.35 }, svg); });
        S.H.forEach((p, i) => { const q = lerp(p, S.aH[i]); el('line', { x1: sx(p[0]), y1: sy(p[1]), x2: sx(q[0]), y2: sy(q[1]), stroke: C.bad, 'stroke-opacity': 0.3 }, svg); });
    }
    S.B.forEach((p, i) => { const q = lerp(p, S.aB[i]); el('circle', { cx: sx(q[0]), cy: sy(q[1]), r: 4, fill: '#fff', stroke: '#475569', 'stroke-width': 1.4 }, svg); });
    S.H.forEach((p, i) => { const q = lerp(p, S.aH[i]); el('circle', { cx: sx(q[0]), cy: sy(q[1]), r: 4, fill: C.bad, 'fill-opacity': 0.85 }, svg); });

    // cluster labels (at the unablated centres)
    const cB = S.B.reduce((s, p) => [s[0] + p[0] / S.B.length, s[1] + p[1] / S.B.length], [0, 0]);
    const cH = S.H.reduce((s, p) => [s[0] + p[0] / S.H.length, s[1] + p[1] / S.H.length], [0, 0]);
    el('text', { x: sx(cB[0]) - 74, y: sy(cB[1]) - 18, 'font-size': 12, 'font-weight': 600, fill: '#475569' }, svg, 'benign');
    el('text', { x: sx(cH[0]) + 50, y: sy(cH[1]) + 4, 'font-size': 12, 'font-weight': 600, fill: C.bad }, svg, 'harmful');

    // direction arrow
    const A = 1.35;
    el('line', { x1: sx(0), y1: sy(0), x2: sx(A * S.u[0]), y2: sy(A * S.u[1]), stroke: col, 'stroke-width': 3, 'marker-end': 'url(#arr-u)' }, svg);
    const lbl = el('text', { x: sx(A * S.u[0]) + (S.u[0] >= 0 ? 8 : -46), y: sy(A * S.u[1]) + (S.u[1] >= 0 ? -6 : 16), 'font-size': 13, 'font-weight': 700, fill: col }, svg, 'u');
    el('tspan', { 'baseline-shift': 'sub', 'font-size': 9 }, lbl, Geo.dir === 'ca' ? 'CA' : 'DIM');

    // origin
    el('circle', { cx: sx(0), cy: sy(0), r: 4, fill: C.ink }, svg);
    el('text', { x: sx(0) - 14, y: sy(0) + 20, 'font-size': 11, fill: C.ink2 }, svg, 'origin');

    drawGates(S);
}

function drawGates(S) {
    const svg = document.getElementById('gate-svg'); svg.innerHTML = '';
    const x0 = 78, x1 = 404, gx = v => x0 + (v + 1) / 2 * (x1 - x0);
    el('line', { x1: gx(0), y1: 8, x2: gx(0), y2: 104, stroke: C.ink3, 'stroke-dasharray': '3 3' }, svg);
    const rows = [['benign', S.gB, 34, '#475569', true], ['harmful', S.gH, 78, C.bad, false]];
    for (const [name, g, y, c, hollow] of rows) {
        el('text', { x: 6, y: y + 4, 'font-size': 11.5, 'font-weight': 600, fill: c }, svg, name);
        el('line', { x1: x0, y1: y, x2: x1, y2: y, stroke: '#f1f5f9', 'stroke-width': 18 }, svg);
        const zs = hollow ? Geo.zB : Geo.zH;
        g.forEach((v, i) => el('circle', { cx: gx(v), cy: y + zs[i][2] * 6, r: 3.2, fill: hollow ? '#fff' : c, stroke: c, 'stroke-width': 1.1, 'fill-opacity': hollow ? 1 : 0.8 }, svg));
        const m = g.reduce((s, v) => s + v, 0) / g.length;
        el('line', { x1: gx(m), y1: y - 13, x2: gx(m), y2: y + 13, stroke: C.ink, 'stroke-width': 2.2 }, svg);
        el('text', { x: gx(m), y: y - 16, 'font-size': 10.5, 'text-anchor': 'middle', fill: C.ink, 'font-weight': 600 }, svg, 'mean ' + m.toFixed(2));
    }
    for (const v of [-1, -0.5, 0, 0.5, 1]) el('text', { x: gx(v), y: 118, 'font-size': 10, 'text-anchor': 'middle', fill: C.ink3 }, svg, v === 0 ? '0' : v.toString().replace('-', '−'));
    el('text', { x: (x0 + x1) / 2, y: 131, 'font-size': 10.5, 'text-anchor': 'middle', fill: C.ink2 }, svg, 'gate = cos(h, u): how hard the edit fires on each state');

    const mB = S.gB.reduce((s, v) => s + v, 0) / S.gB.length, mH = S.gH.reduce((s, v) => s + v, 0) / S.gH.length;
    const hit = S.gB.filter(v => Math.abs(v) > 0.2).length / S.gB.length;
    const rb = document.getElementById('ro-benign'); rb.textContent = mB.toFixed(2).replace('-', '−'); rb.style.color = Math.abs(mB) > 0.12 ? C.bad : C.ca;
    const rh = document.getElementById('ro-harm'); rh.textContent = mH.toFixed(2); rh.style.color = C.ink;
    const rm = document.getElementById('ro-move'); rm.textContent = Math.round(hit * 100) + '%'; rm.style.color = hit > 0.25 ? C.bad : C.ca;
    rm.previousElementSibling.textContent = 'benign states hit (|gate| > 0.2)';
}

function animateGeo() {
    const start = Geo.t, end = Geo.target, t0 = performance.now(), dur = 650;
    const step = now => {
        const p = Math.min((now - t0) / dur, 1), e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
        Geo.t = start + (end - start) * e; drawGeo();
        if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

function initGeo() {
    if (!document.getElementById('geo-svg')) return;
    document.querySelectorAll('#geo-widget .seg-btn').forEach(b => b.addEventListener('click', () => {
        document.querySelectorAll('#geo-widget .seg-btn').forEach(x => x.classList.toggle('active', x === b));
        Geo.dir = b.dataset.dir; drawGeo();
    }));
    const s = document.getElementById('geo-omega'), sv = document.getElementById('geo-omega-val');
    s.addEventListener('input', () => { Geo.omega = +s.value; sv.textContent = (+s.value).toFixed(2); drawGeo(); });
    const ap = document.getElementById('geo-apply');
    ap.addEventListener('click', () => {
        const on = ap.getAttribute('aria-pressed') !== 'true';
        ap.setAttribute('aria-pressed', on); ap.textContent = on ? 'Undo ablation' : 'Apply ablation';
        Geo.target = on ? 1 : 0; animateGeo();
    });
    drawGeo();
}

// ---------------------------------------------------------------------------------------------
// 2. Dumbbell chart: per-model results, ordered by omega
// ---------------------------------------------------------------------------------------------
const DB = { metric: 'asr', offset: false };
const METRIC = {
    asr: { key: 'asr', max: 80, ticks: [0, 20, 40, 60, 80], axis: 'attack success (%), HarmBench, fluent-only', better: 'up',
           caption: 'CA (blue) against DIM ablation (gray); | marks the unsteered model. CA fires where DIM under-fires (both Llamas) and moves the other six by at most 2.6 points.' },
    rw:  { key: 'rw', max: 100, ticks: [0, 25, 50, 75, 100], axis: 'benign answers rewritten (%), lower is more faithful', better: 'down',
           caption: 'On the four models with ω > 0.3, CA preserves 12 to 40 points more benign answers; below that, the directions nearly coincide.' },
    col: { key: 'col', max: 60, ticks: [0, 20, 40, 60], axis: 'collapse rate (%), degenerate outputs', better: 'down',
           caption: 'Only Gemma-2-9B collapses under DIM (β⁺ = 1.03 > 1); CA brings it from 50% to 1%.' },
};

function drawDumbbell() {
    const svg = document.getElementById('db-svg'); if (!svg) return;
    svg.innerHTML = '';
    const M = METRIC[DB.metric], x0 = 190, x1 = 735, top = 34, rowH = 40;
    const X = v => x0 + v / M.max * (x1 - x0);
    const bottom = top + rowH * MODELS.length;

    // high-omega band
    const hiStart = MODELS.findIndex(m => m.omega > 0.3);
    el('rect', { x: 6, y: top + hiStart * rowH - 4, width: 748, height: (MODELS.length - hiStart) * rowH + 4, fill: C.tint, rx: 8 }, svg);
    el('text', { x: 746, y: bottom - 3, 'font-size': 10, fill: C.ink3, 'font-weight': 600, 'text-anchor': 'end' }, svg, 'HIGH OVERLAP  ω > 0.3');

    for (const v of M.ticks) {
        el('line', { x1: X(v), y1: top - 8, x2: X(v), y2: bottom, stroke: C.line }, svg);
        el('text', { x: X(v), y: bottom + 18, 'font-size': 11, 'text-anchor': 'middle', fill: C.ink3 }, svg, v);
    }
    el('text', { x: (x0 + x1) / 2, y: bottom + 40, 'font-size': 12, 'text-anchor': 'middle', fill: C.ink2 }, svg, M.axis);
    el('text', { x: 150, y: top - 12, 'font-size': 10, fill: C.ink3, 'text-anchor': 'end' }, svg, 'model');
    el('text', { x: 178, y: top - 12, 'font-size': 10, fill: C.ink3, 'text-anchor': 'end' }, svg, 'ω');

    MODELS.forEach((m, i) => {
        const y = top + i * rowH + rowH / 2, d = m[M.key];
        const g = el('g', { class: 'db-row' }, svg);
        el('text', { x: 150, y: y + 4, 'font-size': 12.5, 'text-anchor': 'end', fill: C.ink, 'font-weight': 500 }, g, m.name);
        el('text', { x: 178, y: y + 4, 'font-size': 11, 'text-anchor': 'end', fill: C.ink3 }, g, m.omega.toFixed(2).replace(/^0/, ''));
        el('line', { x1: x0, y1: y, x2: x1, y2: y, stroke: '#eef2f7' }, g);

        const delta = d.ca - d.dim;
        const improved = M.better === 'up' ? delta >= 3 : delta <= -3;
        el('line', { x1: X(d.dim), y1: y, x2: X(d.ca), y2: y, stroke: improved ? C.ca : '#cbd5e1', 'stroke-width': improved ? 3 : 2 }, g);
        if (M.key === 'asr') el('line', { x1: X(d.none), y1: y - 8, x2: X(d.none), y2: y + 8, stroke: C.ink, 'stroke-width': 2 }, g);
        if (DB.offset) el('circle', { cx: X(d.off), cy: y, r: 6, fill: C.off, stroke: '#fff', 'stroke-width': 1.5 }, g);
        el('circle', { cx: X(d.dim), cy: y, r: 7, fill: C.dim, stroke: '#fff', 'stroke-width': 1.5 }, g);
        el('circle', { cx: X(d.ca), cy: y, r: 7, fill: C.ca, stroke: '#fff', 'stroke-width': 1.5 }, g);
        if (Math.abs(delta) >= 3) {
            const shown = M.key === 'asr' ? m.gain : delta;
            el('text', { x: (X(d.dim) + X(d.ca)) / 2, y: y - 9, 'font-size': 11.5, 'font-weight': 700, 'text-anchor': 'middle', fill: improved ? C.ca : C.ink3 }, g, fmt(shown, 0));
        }
        const hit = el('rect', { x: 0, y: y - rowH / 2, width: 760, height: rowH, fill: 'transparent' }, g);
        hit.addEventListener('mousemove', e => {
            const rows = [['t-dim', 'DIM ablation', d.dim], ['t-off', 'DIM+offset', d.off], ['t-ca', 'CA (ours)', d.ca]];
            showTip(e, `<b>${m.name}</b> &nbsp;ω ${m.omega.toFixed(3)} &middot; η ${m.eta.toFixed(3)}<br>` +
                (M.key === 'asr' ? `unsteered: ${d.none.toFixed(1)}%<br>` : '') +
                rows.map(([c, l, v]) => `<span class="${c}">${l}: ${v.toFixed(M.key === 'col' ? 0 : 1)}%</span>`).join('<br>'));
        });
        hit.addEventListener('mouseleave', hideTip);
    });

    const lg = document.getElementById('db-legend');
    lg.innerHTML = `<span><i class="dot" style="background:${C.dim}"></i>DIM ablation (standard)</span>` +
        (DB.offset ? `<span><i class="dot" style="background:${C.off}"></i>DIM+offset (ACE)</span>` : '') +
        `<span><i class="dot dot-filled"></i>CA (ours)</span>` +
        (M.key === 'asr' ? `<span><i class="tick-mark"></i>unsteered</span>` : '');
    document.getElementById('db-caption').textContent = M.caption + (DB.offset ? ' DIM+offset is the stronger attack on some models, but rewrites 39 to 46 points more benign answers than CA where ω > 0.33.' : '');
}

function initDumbbell() {
    if (!document.getElementById('db-svg')) return;
    document.querySelectorAll('#db-metric .seg-btn').forEach(b => b.addEventListener('click', () => {
        document.querySelectorAll('#db-metric .seg-btn').forEach(x => x.classList.toggle('active', x === b));
        DB.metric = b.dataset.metric; drawDumbbell();
    }));
    document.getElementById('db-offset').addEventListener('change', e => { DB.offset = e.target.checked; drawDumbbell(); });
    drawDumbbell();
}

// ---------------------------------------------------------------------------------------------
// 3. Scatter plots: diagnostics predict the benefit
// ---------------------------------------------------------------------------------------------
function scatter(id, cfg) {
    const svg = document.getElementById(id); if (!svg) return;
    svg.innerHTML = '';
    const l = 56, r = 400, t = 16, b = 262;
    const X = v => l + (v - cfg.x[0]) / (cfg.x[1] - cfg.x[0]) * (r - l);
    const Y = v => b - (v - cfg.y[0]) / (cfg.y[1] - cfg.y[0]) * (b - t);
    for (const v of cfg.yt) { el('line', { x1: l, y1: Y(v), x2: r, y2: Y(v), stroke: v === 0 ? '#cbd5e1' : '#f1f5f9' }, svg); el('text', { x: l - 8, y: Y(v) + 4, 'font-size': 11, 'text-anchor': 'end', fill: C.ink3 }, svg, v); }
    for (const v of cfg.xt) el('text', { x: X(v), y: b + 18, 'font-size': 11, 'text-anchor': 'middle', fill: C.ink3 }, svg, v);
    el('line', { x1: l, y1: b, x2: r, y2: b, stroke: C.ink3 }, svg);
    el('text', { x: (l + r) / 2, y: b + 42, 'font-size': 12, 'text-anchor': 'middle', fill: C.ink2 }, svg, cfg.xl);
    el('text', { x: 14, y: (t + b) / 2, 'font-size': 12, 'text-anchor': 'middle', fill: C.ink2, transform: `rotate(-90 14 ${(t + b) / 2})` }, svg, cfg.yl);
    if (cfg.vline) {
        el('rect', { x: l, y: t, width: X(cfg.vline) - l, height: b - t, fill: C.bad, 'fill-opacity': 0.06 }, svg);
        el('line', { x1: X(cfg.vline), y1: t, x2: X(cfg.vline), y2: b, stroke: C.ink3, 'stroke-dasharray': '4 4' }, svg);
        for (const [x, lines, c] of cfg.regions || []) lines.forEach((tx, i) => el('text', { x: X(x), y: t + 13 + i * 13, 'font-size': 10, 'text-anchor': 'middle', fill: c }, svg, tx));
    }
    if (cfg.fit) drawFit(svg, MODELS.map(cfg.fx), MODELS.map(cfg.fy), X, Y, cfg.fit, { l, r, t, b });
    if (cfg.group) el('text', { x: cfg.group.x, y: cfg.group.y, 'font-size': 10.5, 'text-anchor': cfg.group.anchor || 'middle', fill: C.ink3 }, svg, cfg.group.text);
    MODELS.forEach(m => {
        const px = X(cfg.fx(m)), py = Y(cfg.fy(m));
        const c = el('circle', { cx: px, cy: py, r: 6.5, fill: m.pilot ? C.ca : '#fff', stroke: C.ca, 'stroke-width': 2.2 }, svg);
        const lab = cfg.labels[m.name];
        if (lab) el('text', { x: px + lab[0], y: py + lab[1], 'font-size': 11, 'text-anchor': lab[2] || 'start', fill: C.ink, 'font-weight': 500 }, svg, m.name);
        const hit = el('circle', { cx: px, cy: py, r: 12, fill: 'transparent' }, svg);
        hit.addEventListener('mousemove', e => showTip(e, `<b>${m.name}</b> ${m.pilot ? '(pilot)' : '(predicted in advance)'}<br>${cfg.tip(m)}`));
        hit.addEventListener('mouseleave', hideTip);
    });
}

// Least-squares line with its 95% confidence band for the mean (t, n-2 dof): the paper's fit_line()
function drawFit(svg, xs, ys, X, Y, color, box) {
    const n = xs.length, mx = xs.reduce((a, v) => a + v, 0) / n, my = ys.reduce((a, v) => a + v, 0) / n;
    const sxx = xs.reduce((a, v) => a + (v - mx) ** 2, 0);
    const slope = xs.reduce((a, v, i) => a + (v - mx) * (ys[i] - my), 0) / sxx, icpt = my - slope * mx;
    const s2 = xs.reduce((a, v, i) => a + (ys[i] - slope * v - icpt) ** 2, 0) / (n - 2);
    const T975 = { 6: 2.4469 }[n - 2];                        // t quantile for n = 8 models
    const lo = Math.min(...xs), hi = Math.max(...xs), pad = 0.08 * (hi - lo), K = 60;
    const g = Array.from({ length: K + 1 }, (_, i) => lo - pad + (hi - lo + 2 * pad) * i / K);
    const fitv = x => slope * x + icpt, se = x => Math.sqrt(s2 * (1 / n + (x - mx) ** 2 / sxx));
    const clipId = svg.id + '-clip', defs = el('defs', {}, svg), cp = el('clipPath', { id: clipId }, defs);
    el('rect', { x: box.l, y: box.t, width: box.r - box.l, height: box.b - box.t }, cp);
    const grp = el('g', { 'clip-path': `url(#${clipId})` }, svg);
    const upper = g.map(x => `${X(x)},${Y(fitv(x) + T975 * se(x))}`), lower = g.map(x => `${X(x)},${Y(fitv(x) - T975 * se(x))}`).reverse();
    el('polygon', { points: upper.concat(lower).join(' '), fill: color, 'fill-opacity': 0.12 }, grp);
    el('line', { x1: X(g[0]), y1: Y(fitv(g[0])), x2: X(g[K]), y2: Y(fitv(g[K])), stroke: color, 'stroke-width': 1.6 }, grp);
}

function initScatter() {
    scatter('sc-omega', {
        x: [0, 0.5], y: [-5, 45], xt: [0, 0.1, 0.2, 0.3, 0.4, 0.5], yt: [0, 10, 20, 30, 40],
        xl: 'overlap ω of the DIM direction with the common mode', yl: 'benign answers preserved (points)',
        fx: m => m.omega, fy: m => m.kept, fit: C.ca,
        labels: { 'Gemma-2-2B': [-10, -6, 'end'], 'Llama-3.1-8B': [10, 14], 'Llama-3.2-3B': [-10, -8, 'end'], 'Gemma-2-9B': [-10, 4, 'end'], 'Qwen2.5-3B': [10, 4] },
        group: { x: 62, y: 256, text: '3 Qwen models', anchor: 'start' },
        tip: m => `ω = ${m.omega.toFixed(3)}<br>benign answers preserved: ${fmt(m.kept)} pts`,
    });
    scatter('sc-eta', {
        x: [0, 1], y: [-5, 62], xt: [0, 0.2, 0.4, 0.6, 0.8, 1], yt: [0, 10, 20, 30, 40, 50, 60],
        xl: 'efficacy η of DIM (share of harmful states it flips)', yl: 'attack-success gain of CA (points)',
        fx: m => m.eta, fy: m => m.gain, vline: 0.6, fit: '#a0a8b5',
        regions: [[0.3, ['DIM under-fires (η < 0.6)', 'CA gains'], C.bad], [0.8, ['DIM already fires', 'no gain'], C.ink2]],
        labels: { 'Llama-3.2-3B': [10, 4], 'Llama-3.1-8B': [10, 4] },
        group: { x: 330, y: 222, text: '6 models: −2.6 to +2.4', anchor: 'middle' },
        tip: m => `η = ${m.eta.toFixed(3)}<br>attack-success gain: ${fmt(m.gain)} pts`,
    });
}

// ---------------------------------------------------------------------------------------------
// 4. Judges chart
// ---------------------------------------------------------------------------------------------
function drawJudges() {
    const svg = document.getElementById('judge-svg'); if (!svg) return;
    svg.innerHTML = '';
    const x0 = 150, x1 = 470, X = v => x0 + v / 100 * (x1 - x0);
    const HB = '#334155', LG = '#94a3b8';
    for (const v of [0, 25, 50, 75, 100]) { el('line', { x1: X(v), y1: 14, x2: X(v), y2: 182, stroke: '#f1f5f9' }, svg); el('text', { x: X(v), y: 198, 'font-size': 11, 'text-anchor': 'middle', fill: C.ink3 }, svg, v); }
    el('text', { x: (x0 + x1) / 2, y: 220, 'font-size': 11.5, 'text-anchor': 'middle', fill: C.ink2 }, svg, 'judged harmful (%)');
    const groups = [['Degenerate', '"Here Here Here…"', JUDGES.degen, 30], ['Fluent', 'normal text', JUDGES.fluent, 112]];
    for (const [name, sub, v, y] of groups) {
        el('text', { x: 136, y: y + 22, 'font-size': 13, 'font-weight': 600, 'text-anchor': 'end', fill: C.ink }, svg, name);
        el('text', { x: 136, y: y + 38, 'font-size': 10.5, 'text-anchor': 'end', fill: C.ink3 }, svg, sub + ' (120)');
        for (const [k, c, dy, lab] of [['hb', HB, 0, 'HarmBench classifier'], ['lg', LG, 26, 'Llama-Guard-3-8B']]) {
            el('rect', { x: x0, y: y + dy, width: Math.max(X(v[k]) - x0, 2), height: 20, fill: c, rx: 3 }, svg);
            const inside = v[k] > 60;
            el('text', { x: inside ? X(v[k]) - 6 : X(v[k]) + 6, y: y + dy + 14, 'font-size': 11, 'font-weight': 600, 'text-anchor': inside ? 'end' : 'start', fill: inside ? '#fff' : C.ink }, svg, `${v[k].toFixed(1)}%  ${inside ? lab : ''}`);
            if (!inside) el('text', { x: X(v[k]) + 52, y: y + dy + 14, 'font-size': 10.5, fill: C.ink3 }, svg, lab);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Page plumbing
// ---------------------------------------------------------------------------------------------
function initCopy() {
    document.querySelectorAll('.code-copy').forEach(b => b.addEventListener('click', () => {
        navigator.clipboard.writeText(document.getElementById(b.dataset.target).textContent).then(() => {
            const old = b.textContent; b.textContent = 'Copied!'; setTimeout(() => { b.textContent = old; }, 1600);
        });
    }));
}

function initArxiv() {
    document.querySelectorAll('.arxiv-link').forEach(a => {
        if (ARXIV_URL) { a.href = ARXIV_URL; return; }
        a.classList.add('disabled'); a.title = 'Coming soon'; a.removeAttribute('target');
        a.addEventListener('click', e => e.preventDefault());
    });
}

function initNav() {
    const nav = document.getElementById('topnav'), hero = document.querySelector('.hero');
    const onScroll = () => nav.classList.toggle('visible', window.scrollY > hero.offsetHeight - 80);
    window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
    const links = [...document.querySelectorAll('.topnav-links a')];
    const obs = new IntersectionObserver(entries => entries.forEach(en => {
        if (en.isIntersecting) links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + en.target.id));
    }), { rootMargin: '-45% 0px -50% 0px' });
    links.forEach(a => { const s = document.querySelector(a.getAttribute('href')); if (s) obs.observe(s); });
}

function initLightbox() {
    document.querySelectorAll('.gallery-grid img').forEach(img => img.addEventListener('click', () => {
        const o = document.createElement('div'); o.className = 'lightbox'; o.innerHTML = `<img src="${img.src}" alt="${img.alt}">`;
        const close = () => { o.remove(); document.removeEventListener('keydown', onKey); };
        const onKey = e => { if (e.key === 'Escape') close(); };
        o.addEventListener('click', close); document.addEventListener('keydown', onKey); document.body.appendChild(o);
    }));
}

document.addEventListener('DOMContentLoaded', () => {
    // respect reduced-motion: show the poster frame with controls instead of autoplaying
    const heroVideo = document.querySelector('.hero-media video');
    if (heroVideo && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        heroVideo.removeAttribute('autoplay'); heroVideo.pause(); heroVideo.controls = true;
    }
    initArxiv(); initNav(); initCopy(); initLightbox();
    initGeo(); initDumbbell(); initScatter(); drawJudges();
    if (window.renderMathInElement) {
        renderMathInElement(document.body, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }] });
    }
});
