/* chart.js — the charting layer, rebuilt on KLineCharts (Apache 2.0).

   KLineCharts is a purpose-built financial charting library with NATIVE multi-
   pane support, so every indicator gets its own real pane that scrolls and zooms
   in perfect lockstep with the candles — no manual chart-syncing, and none of
   the jitter that came with stitching separate lightweight-charts together.

   Design:
     - one candle pane with price overlays (EMAs, VWAP, Bollinger, SuperTrend…)
     - volume and every oscillator / desk study in their own native sub-panes
     - all values are computed server-side and attached to each candle, then
       drawn through tiny custom indicators, so the chart shows EXACTLY the
       numbers the engine used — nothing is recomputed in the browser
     - entry / take-profit / stop lines are price-line overlays

   Public surface (unchanged, so app.js is untouched):
     buildChart(data) · buildStudyMenu(onChange) · markLevels(entry, tp, sl) */

/* TradingView "Dark" palette */
const CHART_BG = "#131722";
const GRID = "#1e222d";
const TEXT = "#d1d4dc";
const AXIS = "#2a2e39";
const CROSS = "#758696";
const UP = "#26a69a", DOWN = "#ef5350";

const STUDIES = {
  /* ---------- your strategy: the pro trend/volatility study set ---------- */
  pro_t3:    { group: "pro", pane: "main", label: "T3 (8, 0.7)", color: "#e0b0ff", width: 2 },
  pro_rf:    { group: "pro", pane: "main", label: "Range Filter (50, 3.0)", color: "#4fd1c5", width: 2 },
  pro_sq:    { group: "pro", pane: "prosq",  label: "Squeeze Momentum", color: "#a78bfa", histogram: true, signed: true },
  pro_cov:   { group: "pro", pane: "provol", label: "Change of Volatility", color: "#f5c542", width: 2 },
  pro_cvol:  { group: "pro", pane: "prochk", label: "Chaikin Volatility", color: "#5b9bf5", width: 2 },
  pro_adx:   { group: "pro", pane: "prodmi", label: "ADX (10)", color: "#f5c542", width: 2 },
  pro_dip:   { group: "pro", pane: "prodmi", label: "DI + (10)", color: "#2ecc8f", width: 1 },
  pro_dim:   { group: "pro", pane: "prodmi", label: "DI − (10)", color: "#f0554f", width: 1 },
  pro_atr_pct:{ group: "pro", pane: "proatr", label: "ATR % (14)", color: "#8b95a8", width: 2 },

  /* ---------- overlays (drawn on the candle pane) ---------- */
  ema9:      { group: "overlay", pane: "main", label: "EMA 9",   color: "#f5c542", width: 1 },
  ema21:     { group: "overlay", pane: "main", label: "EMA 21",  color: "#5b9bf5", width: 2 },
  ema50:     { group: "overlay", pane: "main", label: "EMA 50",  color: "#a78bfa", width: 2 },
  ema200:    { group: "overlay", pane: "main", label: "EMA 200", color: "#f0554f", width: 2 },
  sma20:     { group: "overlay", pane: "main", label: "SMA 20",  color: "#8b95a8", width: 1 },
  vwap:      { group: "overlay", pane: "main", label: "VWAP",    color: "#2ecc8f", width: 2 },
  vwap_up:   { group: "overlay", pane: "main", label: "VWAP +1σ", color: "#1d6b4d", width: 1, dashed: true },
  vwap_dn:   { group: "overlay", pane: "main", label: "VWAP −1σ", color: "#1d6b4d", width: 1, dashed: true },
  bb_up:     { group: "overlay", pane: "main", label: "Bollinger upper", color: "#3f5f8a", width: 1 },
  bb_mid:    { group: "overlay", pane: "main", label: "Bollinger basis", color: "#3f5f8a", width: 1, dashed: true },
  bb_dn:     { group: "overlay", pane: "main", label: "Bollinger lower", color: "#3f5f8a", width: 1 },
  kc_up:     { group: "overlay", pane: "main", label: "Keltner upper", color: "#6b4fa0", width: 1 },
  kc_dn:     { group: "overlay", pane: "main", label: "Keltner lower", color: "#6b4fa0", width: 1 },
  dc_up:     { group: "overlay", pane: "main", label: "Donchian high", color: "#c99a2f", width: 1 },
  dc_dn:     { group: "overlay", pane: "main", label: "Donchian low",  color: "#c99a2f", width: 1 },
  supertrend:{ group: "overlay", pane: "main", label: "SuperTrend", color: "#2ecc8f", width: 2 },
  tenkan:    { group: "overlay", pane: "main", label: "Ichimoku Tenkan", color: "#f5c542", width: 1 },
  kijun:     { group: "overlay", pane: "main", label: "Ichimoku Kijun",  color: "#f0554f", width: 1 },
  senkou_a:  { group: "overlay", pane: "main", label: "Ichimoku Span A", color: "#2ecc8f", width: 1 },
  senkou_b:  { group: "overlay", pane: "main", label: "Ichimoku Span B", color: "#f0554f", width: 1 },
  psar:      { group: "overlay", pane: "main", label: "Parabolic SAR", color: "#d0d0d8", width: 1, dashed: true },

  /* ---------- oscillators, each in its own pane ---------- */
  volume:     { group: "osc", pane: "volume", label: "Volume", color: "#5b9bf5", builtin: "VOL" },
  rsi:        { group: "osc", pane: "rsi",   label: "RSI (14)", color: "#f5c542" },
  stoch_k:    { group: "osc", pane: "stoch", label: "Stochastic %K", color: "#5b9bf5" },
  stoch_d:    { group: "osc", pane: "stoch", label: "Stochastic %D", color: "#f0554f" },
  stochrsi_k: { group: "osc", pane: "stochrsi", label: "Stoch RSI", color: "#a78bfa" },
  macd:       { group: "osc", pane: "macd",  label: "MACD", color: "#5b9bf5" },
  macd_signal:{ group: "osc", pane: "macd",  label: "Signal", color: "#f0554f" },
  macd_hist:  { group: "osc", pane: "macd",  label: "Histogram", color: "#3a7a5c", histogram: true, signed: true },
  adx:        { group: "osc", pane: "adx",   label: "ADX (14)", color: "#f5c542" },
  di_plus:    { group: "osc", pane: "adx",   label: "DI +", color: "#2ecc8f" },
  di_minus:   { group: "osc", pane: "adx",   label: "DI −", color: "#f0554f" },
  cci:        { group: "osc", pane: "cci",   label: "CCI (20)", color: "#a78bfa" },
  williams_r: { group: "osc", pane: "wr",    label: "Williams %R", color: "#5b9bf5" },
  mfi:        { group: "osc", pane: "mfi",   label: "Money Flow", color: "#2ecc8f" },
  atr_pct:    { group: "osc", pane: "atr",   label: "ATR %", color: "#f5c542" },
  obv:        { group: "osc", pane: "obv",   label: "On-Balance Volume", color: "#8b95a8" },
  roc:        { group: "osc", pane: "roc",   label: "Rate of Change", color: "#5b9bf5" },

  /* ---------- proprietary desk studies ---------- */
  cvd:        { group: "desk", pane: "cvd",   label: "Cumulative Volume Delta", color: "#2ecc8f", width: 2 },
  delta:      { group: "desk", pane: "delta", label: "Volume Delta", color: "#5b9bf5", histogram: true, signed: true },
  aggressor:  { group: "desk", pane: "aggr",  label: "Aggressor Imbalance", color: "#f5c542" },
  whale_flow: { group: "desk", pane: "whale", label: "Whale Flow", color: "#a78bfa" },
  whale_share:{ group: "desk", pane: "whale", label: "Whale Share", color: "#6b4fa0" },
  absorption: { group: "desk", pane: "absorb", label: "Absorption", color: "#f0554f" },
  intensity:  { group: "desk", pane: "inten", label: "Trade Intensity", color: "#2ecc8f" },
  oi_chg:     { group: "desk", pane: "oi",    label: "Open Interest Δ%", color: "#f5c542" },
  crowd_gap:  { group: "desk", pane: "crowd", label: "Retail vs Smart Money", color: "#f0554f" },
};

const DEFAULT_ON = ["pro_t3", "pro_rf", "ema21", "ema50", "vwap", "volume", "pro_sq", "cvd"];
const QUICK = ["pro_t3", "pro_rf", "pro_sq", "pro_cvol", "pro_adx",
               "ema21", "ema50", "vwap", "supertrend", "volume", "rsi", "cvd"];

let CHART = null, ACTIVE = new Set(DEFAULT_ON), LAST_DATA = null, ON_CHANGE = null;
let LAST_KLINES = [];   // candles with indicator values attached, for the HUD legend
let HIGHLIGHT = null, HL_TIMER = null, CLICK_WIRED = false;   // click-a-line-to-open state
let PRICE_DP = 2;
let RESIZE_OBS = null;
let SAVED_LEVELS = null, LEVEL_IDS = [];
const REGISTERED = new Set();   // custom indicators only need registering once

/* KLineCharts theme — TradingView dark */
function themeStyles() {
  return {
    grid: { horizontal: { color: GRID }, vertical: { color: GRID } },
    candle: {
      bar: { upColor: UP, downColor: DOWN, upBorderColor: UP, downBorderColor: DOWN,
             upWickColor: UP, downWickColor: DOWN },
      priceMark: {
        last: { text: { color: "#131722" } },
        high: { color: TEXT }, low: { color: TEXT },
      },
      tooltip: { rect: { color: "rgba(19,23,34,0.9)", borderColor: AXIS },
                 text: { color: TEXT } },
    },
    indicator: {
      // native indicator legend is hidden — we draw our own clickable HUD instead
      tooltip: { showRule: "none", text: { color: TEXT } },
      lastValueMark: { show: false },
    },
    xAxis: { axisLine: { color: AXIS }, tickLine: { color: AXIS }, tickText: { color: "#9b9daa" } },
    yAxis: { axisLine: { color: AXIS }, tickLine: { color: AXIS }, tickText: { color: "#9b9daa" } },
    separator: { size: 1, color: AXIS },
    crosshair: {
      horizontal: { line: { color: CROSS, style: "dashed" },
                    text: { color: "#fff", backgroundColor: "#363a45", borderColor: "#363a45" } },
      vertical:   { line: { color: CROSS, style: "dashed" },
                    text: { color: "#fff", backgroundColor: "#363a45", borderColor: "#363a45" } },
    },
  };
}

/* how many decimals the instrument trades in */
function detectPrecision(candles) {
  let dp = 0;
  for (const c of candles.slice(-300)) {
    const s = String(c.close);
    const dot = s.indexOf(".");
    if (dot >= 0) dp = Math.max(dp, s.length - dot - 1);
  }
  return Math.min(Math.max(dp, 1), 8);
}

/* Register a one-line (or one-bar) custom indicator that simply draws the value
   we already attached to each candle under `key`. Colour is baked in from the
   study spec; signed histograms colour green/red by sign like TradingView. */
function registerStudy(key, spec) {
  if (REGISTERED.has(key)) return;
  const isBar = !!spec.histogram;
  klinecharts.registerIndicator({
    name: "S_" + key,
    shortName: spec.label,
    precision: spec.pane === "main" ? PRICE_DP : 4,
    figures: [{
      key: "v",
      title: spec.label + ": ",
      type: isBar ? "bar" : "line",
      styles: (data) => {
        const v = data && data.current && data.current.indicatorData
          ? data.current.indicatorData.v : null;
        const hl = HIGHLIGHT === key;   // this line is the one just clicked
        if (isBar) {
          const col = spec.signed ? (v >= 0 ? UP : DOWN) : spec.color;
          return { color: col, style: "fill" };
        }
        return { color: hl ? "#ffffff" : spec.color,
                 size: (spec.width || 1) + (hl ? 2 : 0),
                 style: spec.dashed ? "dashed" : "solid" };
      },
    }],
    calc: (kd) => kd.map((d) => {
      const v = d[key];
      return { v: (v == null || Number.isNaN(v)) ? null : v };
    }),
  });
  REGISTERED.add(key);
}

/* which sub-panes are needed, in a stable order, so we can size the container */
function activeSubPanes() {
  const seen = [];
  for (const key of ACTIVE) {
    const spec = STUDIES[key];
    if (!spec || spec.pane === "main") continue;
    const id = "pane_" + spec.pane;
    if (!seen.includes(id)) seen.push(id);
  }
  return seen;
}

function buildChart(data) {
  LAST_DATA = data;
  const el = document.getElementById("chart-main");
  const subs = document.getElementById("sub-panes");
  if (subs) subs.innerHTML = "";      // legacy container, unused now

  // tear down any previous chart so a rebuild is clean
  if (RESIZE_OBS) { RESIZE_OBS.disconnect(); RESIZE_OBS = null; }
  if (CHART) { klinecharts.dispose(el); CHART = null; }
  LEVEL_IDS = [];

  PRICE_DP = detectPrecision(data.candles);

  // give the container a height that keeps the candle pane tall while every
  // sub-pane gets a fixed slice
  const nSub = activeSubPanes().length;
  el.style.background = CHART_BG;
  el.style.height = (470 + nSub * 108) + "px";

  CHART = klinecharts.init(el, { styles: themeStyles() });
  CHART.setPriceVolumePrecision(PRICE_DP, 0);

  // attach every server-computed indicator value onto its candle, so the custom
  // indicators can draw them without any browser-side recomputation
  const inds = data.indicators || {};
  const klineData = data.candles.map((c, i) => {
    const o = { timestamp: c.time * 1000, open: c.open, high: c.high,
                low: c.low, close: c.close, volume: c.volume || 0 };
    for (const k in inds) {
      const arr = inds[k];
      o[k] = arr && arr[i] != null ? arr[i] : undefined;
    }
    return o;
  });
  CHART.applyNewData(klineData);
  LAST_KLINES = klineData;

  // draw the active studies
  for (const key of ACTIVE) {
    const spec = STUDIES[key];
    if (!spec) continue;
    if (spec.builtin) {                       // volume uses the native VOL pane
      CHART.createIndicator(spec.builtin, true, { id: "pane_" + spec.pane, height: 96 });
      continue;
    }
    if (!(key in inds)) continue;             // no data for this study
    registerStudy(key, spec);
    if (spec.pane === "main") {
      CHART.createIndicator("S_" + key, true, { id: "candle_pane" });
    } else {
      CHART.createIndicator("S_" + key, true, { id: "pane_" + spec.pane, height: 104 });
    }
  }

  if (SAVED_LEVELS) markLevels(SAVED_LEVELS[0], SAVED_LEVELS[1], SAVED_LEVELS[2]);

  // keep the chart sized to its container
  RESIZE_OBS = new ResizeObserver(() => { if (CHART) CHART.resize(); });
  RESIZE_OBS.observe(el);

  renderLegend();
  renderQuickBar();

  // clickable HUD legend over the chart, with live values on the crosshair
  buildHud();
  CHART.subscribeAction("onCrosshairChange", (data) => {
    let k = null;
    if (data && typeof data.dataIndex === "number" && LAST_KLINES[data.dataIndex]) {
      k = LAST_KLINES[data.dataIndex];
    } else if (data && data.kLineData) {
      k = data.kLineData;
    }
    updateHudValues(k || LAST_KLINES[LAST_KLINES.length - 1]);
  });

  // click a drawn line on the candle pane → highlight it and open its card
  if (!CLICK_WIRED) {
    CLICK_WIRED = true;
    el.addEventListener("click", (ev) => {
      const key = chartLineAt(ev.clientX, ev.clientY);
      if (key) { highlightLine(key); showIndicatorInfo(key); }
    });
  }
}

/* which main-pane overlay line (if any) sits under the click, within a few px */
function chartLineAt(clientX, clientY) {
  if (!CHART) return null;
  const el = document.getElementById("chart-main");
  const rect = el.getBoundingClientRect();
  const x = clientX - rect.left, y = clientY - rect.top;
  let di;
  try {
    const c = CHART.convertFromPixel({ x, y }, { paneId: "candle_pane" });
    di = Array.isArray(c) ? (c[0] || {}).dataIndex : (c || {}).dataIndex;
  } catch (e) { return null; }
  if (di == null || !LAST_KLINES[di]) return null;
  let best = null, bestD = 8;   // px tolerance
  for (const key of ACTIVE) {
    const s = STUDIES[key];
    if (!s || s.pane !== "main" || s.builtin) continue;
    const v = LAST_KLINES[di][key];
    if (v == null || Number.isNaN(v)) continue;
    let py;
    try {
      const p = CHART.convertToPixel(
        { dataIndex: di, timestamp: LAST_KLINES[di].timestamp, value: v },
        { paneId: "candle_pane" });
      py = Array.isArray(p) ? (p[0] || {}).y : (p || {}).y;
    } catch (e) { continue; }
    if (py == null) continue;
    const d = Math.abs(py - y);
    if (d < bestD) { bestD = d; best = key; }
  }
  return best;
}

/* thicken + whiten the clicked line briefly, and flash its HUD row */
function highlightLine(key) {
  HIGHLIGHT = key;
  try { CHART.overrideIndicator({ name: "S_" + key }); } catch (e) { /* redraw */ }
  const row = document.querySelector(`.hud-row[data-k="${key}"]`);
  if (row) row.classList.add("hud-hot");
  clearTimeout(HL_TIMER);
  HL_TIMER = setTimeout(() => {
    const k = HIGHLIGHT; HIGHLIGHT = null;
    try { if (k) CHART.overrideIndicator({ name: "S_" + k }); } catch (e) { /* redraw */ }
    document.querySelectorAll(".hud-row.hud-hot").forEach((r) => r.classList.remove("hud-hot"));
  }, 1600);
}

/* how many decimals a HUD value gets */
function fmtHudVal(v, spec) {
  if (v == null || Number.isNaN(v)) return "—";
  if (spec.pane === "main") {
    return Number(v).toLocaleString(undefined,
      { minimumFractionDigits: PRICE_DP, maximumFractionDigits: PRICE_DP });
  }
  const a = Math.abs(v);
  const dp = a >= 100 ? 0 : a >= 1 ? 2 : 4;
  return Number(v).toFixed(dp);
}

/* build the on-chart clickable legend (name + colour + live value) */
function buildHud() {
  const hud = document.getElementById("chart-hud");
  if (!hud) return;
  const last = LAST_KLINES[LAST_KLINES.length - 1] || {};
  const rows = [...ACTIVE]
    .filter((k) => STUDIES[k] && !STUDIES[k].builtin)   // volume has no attached series
    .map((k) => {
      const s = STUDIES[k];
      return `<div class="hud-row" data-k="${k}" role="button" tabindex="0" ` +
             `title="Click for details, signals & how to trade it">` +
             `<i style="background:${s.color}"></i>` +
             `<span class="hud-name">${s.label}</span>` +
             `<span class="hud-val" data-hv="${k}">${fmtHudVal(last[k], s)}</span></div>`;
    });
  hud.innerHTML = rows.join("");
}

function updateHudValues(kline) {
  if (!kline) return;
  for (const k of ACTIVE) {
    if (!STUDIES[k] || STUDIES[k].builtin) continue;
    const el = document.querySelector(`.hud-val[data-hv="${k}"]`);
    if (el) el.textContent = fmtHudVal(kline[k], STUDIES[k]);
  }
}

/* entry / take-profit / stop as horizontal price-line overlays */
function markLevels(entry, tp, sl) {
  SAVED_LEVELS = [entry, tp, sl];
  if (!CHART) return;
  LEVEL_IDS.forEach((id) => CHART.removeOverlay(id));
  LEVEL_IDS = [];
  const line = (value, color) => {
    if (value == null) return;
    const id = CHART.createOverlay({
      name: "priceLine",
      points: [{ value }],
      lock: true,
      styles: { line: { color, style: "dashed", size: 1 },
                text: { color: "#fff", backgroundColor: color } },
    });
    if (id) LEVEL_IDS.push(id);
  };
  line(entry, "#e6e8ea");
  line(tp, UP);
  line(sl, DOWN);
}

/* the top legend row — names of the active price-pane overlays */
function renderLegend() {
  const el = document.getElementById("chart-legend");
  if (!el) return;
  // every active study is a clickable, colour-coded chip — click one to open its
  // full detail card (overview, interpretation, long/short signals, formula)
  el.innerHTML = [...ACTIVE]
    .filter((k) => STUDIES[k])
    .map((k) => `<span class="lg" data-k="${k}" tabindex="0" role="button" ` +
                `title="Click for details, signals & how to trade it">` +
                `<i style="background:${STUDIES[k].color}"></i>${STUDIES[k].label}</span>`)
    .join("");
}

/* the one-click quick toggles above the chart */
function renderQuickBar() {
  const bar = document.getElementById("quick-bar");
  if (!bar) return;
  bar.innerHTML = QUICK.map((k) => {
    const s = STUDIES[k];
    if (!s) return "";
    return `<button class="qchip ${ACTIVE.has(k) ? "on" : ""}" data-k="${k}">
      <i style="background:${s.color}"></i>${s.label}</button>`;
  }).join("");
  bar.querySelectorAll(".qchip").forEach((b) =>
    b.addEventListener("click", () => {
      const k = b.dataset.k;
      if (ACTIVE.has(k)) ACTIVE.delete(k); else ACTIVE.add(k);
      if (ON_CHANGE) ON_CHANGE();
    }));
}

/* the full studies picker (the modal grouped into overlays / oscillators / desk) */
function buildStudyMenu(onChange) {
  ON_CHANGE = onChange;
  const groups = { pro: "std-pro", overlay: "std-overlays", osc: "std-oscillators", desk: "std-desk" };
  for (const g of Object.values(groups)) {
    const el = document.getElementById(g);
    if (el) el.innerHTML = "";
  }
  for (const [key, spec] of Object.entries(STUDIES)) {
    const host = document.getElementById(groups[spec.group]);
    if (!host) continue;
    const row = document.createElement("label");
    row.className = "std-row";
    row.innerHTML =
      `<input type="checkbox" ${ACTIVE.has(key) ? "checked" : ""}>` +
      `<span class="std-dot" style="background:${spec.color}"></span>` +
      `<span class="std-name">${spec.label}</span>` +
      `<button type="button" class="std-info" data-k="${key}" ` +
      `title="How to use — where to buy / sell">?</button>`;
    host.appendChild(row);
    row.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) ACTIVE.add(key); else ACTIVE.delete(key);
      onChange();
    });
  }
  renderQuickBar();
}
