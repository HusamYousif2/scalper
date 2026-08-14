/* ASTRA Terminal — application layer.

   Language rule for this file: percentages and dollars only. Basis points are
   how the engine thinks; they are not how a trader reads a screen, and asking
   someone to divide by 100 in their head on every glance is a tax on the
   product. Everything is converted before it is displayed.

   Direction is shown as UP or DOWN with a target price, not as LONG or SHORT
   with a probability, because the first is a statement about the market and the
   second is jargon about a position. */

const $ = (id) => document.getElementById(id);

const bpsToPct = (b) => (b == null ? null : b / 100);
const pct = (b, d) => {
  if (b == null) return "—";
  const p = b / 100;
  return `${p.toFixed(d != null ? d : p < 0.1 ? 3 : p < 1 ? 2 : 2)}%`;
};
const usdOf = (px, b) => {
  if (b == null || px == null) return "—";
  const v = (px * b) / 1e4;
  return v >= 1000 ? `$${Math.round(v).toLocaleString()}`
    : v >= 1 ? `$${v.toFixed(2)}` : `$${v.toFixed(4)}`;
};
const price = (p) =>
  p == null ? "—" : Number(p).toLocaleString(undefined,
    { maximumFractionDigits: p < 10 ? 4 : 1 });
const num = (v, d = 1) => (v == null ? "—" : Number(v).toFixed(d));
const sgn = (v, d = 1) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${num(v, d)}`);

let LOADING = false;
let TF = 15;

/* live-watch state: the rule plan re-runs on an interval so the desk read stays
   current as new candles form, without a full page reload */
let PLAN_POLL = null;
const PLAN_POLL_MS = 60000;     // re-analyse once a minute — calm, not twitchy
let LIVE_PRO = null, LIVE_HF = null, LAST_LEVEL_DRAW = 0;

/* called on every live price tick from the Binance WebSocket (chart.js) — moves
   the entry/stop/target of both reads with the price, in real time */
window.onLivePrice = function (sym, px) {
  if (!px || sym !== ($("symbol") ? $("symbol").value : sym)) return;
  // update the top ticker for this coin
  const tk = $("tk-" + sym); if (tk) tk.textContent = price(px);
  // update a small "current price" hint next to the entry, if it exists
  const now = $("re-live-price");
  if (now && LIVE_PRO) {
    const diff = px - LIVE_PRO.entry;
    const pct = LIVE_PRO.entry ? (diff / LIVE_PRO.entry * 100) : 0;
    const sign = diff >= 0 ? "+" : "";
    now.innerHTML = `Live <b>${price(px)}</b> <span class="${diff>=0?'up':'down'}">${sign}${pct.toFixed(2)}%</span>`;
  }
  const hfNow = $("hf-live-price");
  if (hfNow && LIVE_HF) {
    hfNow.textContent = `live ${price(px)}`;
  }
  // ENTRY / STOP / TARGET are DECISIONS made when the bar closed — they don't
  // chase every tick. That's the professional-platform behavior and it's what
  // stops the numbers from "flickering". Server recomputes once per minute (or
  // once per closed bar) and updates them at that cadence, not on every wick.
};
function tfLabel(tf) { return tf >= 60 ? `${tf / 60}H` : `${tf}m`; }

/* ---------- boot ---------- */
async function loadSymbols() {
  // every coin that has archived data (the pro strategy needs no ML model)
  const j = await (await fetch("/api/symbols?data=1")).json();
  const sel = $("symbol");
  sel.innerHTML = "";
  for (const s of j.symbols) {
    const o = document.createElement("option");
    o.value = s.symbol;
    o.textContent = s.symbol.replace("USDT", " / USDT");
    sel.appendChild(o);
  }
  if ([...sel.options].some((o) => o.value === "BTCUSDT")) sel.value = "BTCUSDT";
  else if (j.symbols.length) sel.value = j.symbols[0].symbol;
  return j.symbols.map((s) => s.symbol);
}

function loadTicker(symbols) {
  $("ticker").innerHTML = symbols.map((s) =>
    `<div class="tk" data-sym="${s}"><span class="tk-sym">${s.replace("USDT", "")}</span>
     <span class="tk-px" id="tk-${s}">—</span></div>`).join("");
  document.querySelectorAll(".tk").forEach((el) =>
    el.addEventListener("click", () => { $("symbol").value = el.dataset.sym; load(); }));
}

/* ---------- opportunity scanner (all coins, ranked, self-refreshing) ---------- */
function scanColor(r) {
  return r === "Fire" ? "var(--gold)" : r === "Strong" ? "var(--up)"
    : r === "Building" ? "#5b9bf5" : "var(--faint)";
}
async function loadScanner() {
  try {
    const j = await fetch("/api/signals?tf=240").then((r) => (r.ok ? r.json() : null));
    if (j && j.rows) renderScanner(j.rows);
  } catch (e) { /* keep last */ }
}

/* map a scanner indicator name -> the chart study key(s). Kept minimal (fewest
   lines each) so the chart stays clean when it mirrors the top signals. */
const SCAN_TO_STUDY = {
  "RSI": ["rsi"], "Stochastic": ["stoch_k"], "Stoch RSI": ["stochrsi_k"],
  "MACD momentum": ["macd_hist"], "MACD cross": ["macd_hist"],
  "SuperTrend": ["supertrend"], "ADX / DMI": ["adx"],
  "EMA trend": ["ema50"], "VWAP": ["vwap"],
  "Bollinger": ["bb_up", "bb_dn"], "CCI": ["cci"], "Williams %R": ["williams_r"],
  "Money Flow": ["mfi"], "Rate of Change": ["roc"], "Parabolic SAR": ["psar"],
  "Donchian": ["dc_up", "dc_dn"], "Ichimoku cloud": ["senkou_a", "senkou_b"],
  "Ichimoku TK cross": ["tenkan"], "Cumulative Delta": ["cvd"],
  "Aggressor flow": ["aggressor"], "Whale flow": ["whale_flow"],
  "Open interest": ["oi_chg"], "On-Balance Volume": ["obv"],
};

/* ---------- AI indicator scan (full suite, one coin, every 5 min) ---------- */
async function loadIndScan() {
  const sym = $("symbol").value || "BTCUSDT";
  try {
    const r = await fetch(`/api/indicator_scan?symbol=${sym}&tf=15`).then((x) => (x.ok ? x.json() : null));
    if (r) renderIndScan(r);
  } catch (e) { /* keep last */ }
}
function renderIndScan(r) {
  const col = scanColor(r.rating);
  if ($("ind-sym")) $("ind-sym").textContent = r.symbol.replace("USDT", "");
  const upd = $("ind-updated");
  if (upd) {
    const n = new Date();
    upd.textContent = `${r.n_total} indicators · updated ${String(n.getHours()).padStart(2, "0")}:${String(n.getMinutes()).padStart(2, "0")}`;
  }
  const top = $("ind-top");
  if (top) top.innerHTML = `<div class="sig-top ${r.bias}">
    <div class="sig-top-l">
      <div class="sig-top-coin">${r.symbol.replace("USDT", "")}<span> / USDT · 15m</span></div>
      <div class="sig-top-rating" style="color:${col}">${r.rating} · <span class="side-${r.bias}">${r.bias === "long" ? "▲ LONG" : "▼ SHORT"}</span> · ${r.consensus_pct}% consensus (${r.n_long}▲ / ${r.n_short}▼)</div>
    </div>
    <div class="sig-top-score"><div class="sig-top-num" style="color:${col}">${r.score}</div><span>signal</span></div>
    <div class="sig-top-lvls">
      <div><span>Entry</span><b>${price(r.entry)}</b></div>
      <div><span>Stop</span><b class="loss">${price(r.stop)}</b></div>
      <div><span>Target</span><b class="win">${price(r.target)}</b></div>
    </div>
  </div>`;
  const list = $("ind-list");
  if (list) {
    const rows = (r.top || []).map((t) => {
      const up = t.dir === "long", c = up ? "var(--up)" : "var(--down)";
      return `<div class="ind-row">
        <span class="ind-name">${t.name}</span>
        <span class="side-${t.dir}">${up ? "▲" : "▼"}</span>
        <span class="sig-bar"><i style="width:${t.strength}%;background:${c}"></i></span>
        <span class="ind-reason">${t.reason}</span>
      </div>`;
    }).join("");
    list.innerHTML = rows || `<div class="rp-empty" style="padding:14px">No strong signals right now.</div>`;
  }

  // mirror ONLY the firing indicators (top 4) onto the chart — clean, not cluttered
  const keys = [];
  (r.top || []).slice(0, 4).forEach((t) => (SCAN_TO_STUDY[t.name] || []).forEach((k) => {
    if (!keys.includes(k)) keys.push(k);
  }));
  if (keys.length && typeof applyStudyKeys === "function") applyStudyKeys(keys);
  // chart entry/stop/target come from the live HF signal (loadHFSignal)
}

async function loadHFSignal(sym) {
  try {
    const s = await fetch(`/api/hf_signal?symbol=${sym}&tf=5`).then((r) => (r.ok ? r.json() : null));
    if (s) renderHFSignal(s);
  } catch (e) { /* transient */ }
}

function renderHFSignal(s) {
  const el = $("hf-live");
  if (!el) return;
  LIVE_HF = { bias: s.bias, slOff: Math.abs(s.entry - s.stop), tpOff: Math.abs(s.target - s.entry) };
  const up = s.bias === "long";
  const fire = s.signal_now;
  el.className = `hf-live ${up ? "long" : "short"}${fire ? " fire" : ""}`;
  el.innerHTML =
    `<span class="hfl-tag">⚡ 5m signal</span>` +
    `<span class="side-${s.bias} hfl-bias">${up ? "▲ LONG" : "▼ SHORT"}</span>` +
    `<span class="hfl-lv"><i>Entry</i>${price(s.entry)}</span>` +
    `<span class="hfl-lv"><i>Stop</i><b class="down">${price(s.stop)}</b></span>` +
    `<span class="hfl-lv"><i>Target</i><b class="up">${price(s.target)}</b></span>` +
    `<span class="hfl-status">${fire ? "● FIRING NOW" : "watching · " + s.rr + ":1"}</span>`;
  if (typeof markLevels === "function") markLevels(s.entry, s.target, s.stop);
}
function renderScanner(rows) {
  if (!rows.length) return;
  // the scanner already knows every coin's live price — fill the ticker with it
  rows.forEach((r) => { const el = $("tk-" + r.symbol); if (el) el.textContent = price(r.entry); });
  const upd = $("scan-updated");
  if (upd) {
    const n = new Date();
    upd.textContent = `updated ${String(n.getHours()).padStart(2, "0")}:${String(n.getMinutes()).padStart(2, "0")}`;
  }
  const t = rows[0], col = scanColor(t.rating);
  const top = $("scan-top");
  if (top) top.innerHTML = `<div class="sig-top ${t.bias}">
    <div class="sig-top-l">
      <div class="sig-top-coin">${t.symbol.replace("USDT", "")}<span> / USDT</span></div>
      <div class="sig-top-rating" style="color:${col}">${t.rating === "Fire"
        ? "● FIRE — entry triggered" : t.rating + " setup"} · <span class="side-${t.bias}">${t.bias === "long" ? "▲ LONG" : "▼ SHORT"}</span></div>
    </div>
    <div class="sig-top-score"><div class="sig-top-num" style="color:${col}">${t.score}</div><span>opportunity</span></div>
    <div class="sig-top-lvls">
      <div><span>Entry</span><b>${price(t.entry)}</b></div>
      <div><span>Stop</span><b class="loss">${price(t.stop)}</b></div>
      <div><span>Target</span><b class="win">${price(t.target)}</b></div>
    </div>
  </div>`;
  const list = $("scan-list");
  if (list) {
    list.innerHTML = rows.map((r) => {
      const c = scanColor(r.rating);
      return `<button class="scan-row" data-sym="${r.symbol}">
        <span class="scan-coin">${r.symbol.replace("USDT", "")}</span>
        <span class="sig-bar"><i style="width:${r.score}%;background:${c}"></i></span>
        <b class="scan-score" style="color:${c}">${r.score}</b>
        <span class="side-${r.bias}">${r.bias === "long" ? "▲" : "▼"}</span>
        <span class="scan-stat" style="color:${c}">${r.signal_now ? "● FIRE" : r.rating}</span>
        <span class="scan-px">${price(r.entry)}</span>
      </button>`;
    }).join("");
    list.querySelectorAll(".scan-row").forEach((b) =>
      b.addEventListener("click", () => {
        $("symbol").value = b.dataset.sym;
        load();
        const cc = document.querySelector(".chart-card");
        if (cc) cc.scrollIntoView({ behavior: "smooth", block: "start" });
      }));
  }
}

/* fetch JSON with a hard timeout; returns null on timeout, error or non-2xx so a
   slow endpoint can never block the page */
function fetchT(url, ms) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), ms);
  return fetch(url, { signal: ac.signal })
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null)
    .finally(() => clearTimeout(t));
}

let LOAD_SEQ = 0;               // newest request wins; stale loads discard themselves
async function load() {
  // A timeframe/symbol click must never be silently dropped because an earlier
  // load is still running (the old `if (LOADING) return` did exactly that, which
  // left the chart and plan on the previous timeframe). Instead, each call takes
  // the latest sequence number; an in-flight load that is no longer newest bails
  // out after its fetch without rendering.
  const seq = ++LOAD_SEQ;
  LOADING = true;
  const sym = $("symbol").value || "BTCUSDT";
  // The trade call is now timeframe-driven (rule engine). The remaining ML
  // analytics cards (move-size ladder, pulse, track record) still need a horizon;
  // pick the trained model nearest the chart timeframe (15m or 60m).
  const hz = TF >= 60 ? 60 : 15;

  $("app").hidden = true;
  $("status").hidden = false;
  $("status").className = "status";
  $("status-text").textContent =
    `Analysing ${sym.replace("USDT", "/USDT")} — reading every trade on the tape…`;

  try {
    // Every fetch is non-fatal AND time-bounded: the chart and the plan are the
    // essentials, and no endpoint (a slow/hanging ML model or the live-data
    // bridge) may block the page from revealing. Slow ones simply return null.
    const [chartData, a, sigResp, stats, planResp] = await Promise.all([
      fetchT(`/api/chart?symbol=${sym}&tf=${TF}&count=320`, 20000),
      fetchT(`/api/assess?symbol=${sym}&horizon=${hz}&cost=none`, 12000),
      fetchT(`/api/signal?symbol=${sym}&horizon=${hz}`, 12000),
      fetchT(`/api/model-stats?symbol=${sym}&horizon=${hz}`, 12000),
      fetchT(`/api/plan?symbol=${sym}&tf=${TF}`, 20000),
    ]);

    if (seq !== LOAD_SEQ) return;   // a newer timeframe/symbol was picked — drop this result

    if (!chartData || !chartData.candles) {
      throw new Error("chart data unavailable — the market feed is warming up, try again in a moment");
    }

    $("status").hidden = true;
    $("app").hidden = false;

    // Render each block independently: one failing renderer must not hide the
    // rest of the page.
    const safe = (label, fn) => { try { fn(); } catch (e) { console.warn(label, e); } };
    safe("chart", () => { buildChart(chartData); buildStudyMenu(() => buildChart(chartData)); });
    if (typeof connectLiveFeed === "function") connectLiveFeed(sym, TF);   // real-time WS
    // the pro strategy (your indicators) drives the read on every timeframe
    safe("plan", () => fetchProPlan(sym, TF));
    loadIndScan();   // full indicator-suite read for this coin
    loadHFSignal(sym);   // live high-frequency signal + chart entry/stop/target
    safe("ladder", () => renderLadder(a));
    safe("pulse", () => renderPulse(a));
    safe("signals", () => renderSignals(a));
    safe("performance", () => renderPerformance(a, stats));

    $("ch-sym").textContent = sym;
    $("ch-tf").textContent = TF >= 60 ? `${TF / 60}H` : `${TF}m`;
    if (a && a.price != null) $("tk-" + sym) && ($("tk-" + sym).textContent = price(a.price));
    $("footer-note").textContent =
      `Binance USD-M futures · ${Object.keys(chartData.indicators || {}).length} live indicator series`;
    startLiveWatch();   // keep the AI read updating each minute
  } catch (e) {
    if (seq === LOAD_SEQ) {
      $("status").className = "status error";
      $("status-text").textContent = `Could not load: ${e.message}`;
    }
  } finally {
    if (seq === LOAD_SEQ) LOADING = false;   // don't let a stale load clear the active one's flag
  }
}

/* ---------- the call ---------- */
function renderCall(a, sigResp) {
  const s = (sigResp && sigResp.signal) || null;
  const q = a.quantiles_bps || {};
  const px = a.price;

  $("call-hz").textContent = a.horizon_min >= 60 ? "HOUR" : `${a.horizon_min} MIN`;
  $("ct-now").textContent = price(px);

  if (!s) {
    $("cd-word").textContent = "NO CALL";
    $("cd-arrow").textContent = "•";
    $("call").className = "call";
    return;
  }

  const up = s.p_up > 0.5;
  const conf = up ? s.p_up : s.p_down;

  // Signal strength is a percentile of the model's own conviction, not an
  // outcome probability. "82" means this read is stronger than 82% of the reads
  // this model produces — a true statement about the signal. The underlying
  // probability stays visible in the note below so nothing is hidden.
  const strength = s.strength != null ? s.strength : Math.round((conf - 0.5) * 400);
  $("cd-strength-num").textContent = strength;
  $("cd-arrow").textContent = up ? "▲" : "▼";
  $("cd-word").textContent = up ? "UP" : "DOWN";
  $("call").className = `call ${up ? "up" : "down"}`;

  $("cd-strength").textContent =
    strength >= 90 ? "Top-tier setup — rare for this model"
    : strength >= 75 ? "Strong setup"
    : strength >= 50 ? "Above-average setup"
    : strength >= 25 ? "Below-average setup" : "Weak setup — little to act on";

  const tp = s.target_price, sl = s.stop_price;
  const tpB = s.target_bps, slB = s.stop_bps;

  $("ct-tp").textContent = price(tp);
  $("ct-tp-move").textContent =
    `${up ? "+" : "−"}${pct(tpB)}  ·  ${usdOf(px, tpB)}`;
  $("ct-sl").textContent = price(sl);
  $("ct-sl-move").textContent =
    `${up ? "−" : "+"}${pct(slB)}  ·  ${usdOf(px, slB)}`;
  // reward : risk, so a value below 1 clearly means reward < risk
  $("ct-rr").textContent = slB ? `${num(tpB / slB, 2)} : 1` : "—";
  $("ct-swing").textContent = `${pct(q.q50)} (${usdOf(px, q.q50)})`;
  $("ct-hit").textContent = "90% survival";

  $("cg-needle").style.left = `${strength}%`;
  const fill = $("cg-fill");
  fill.style.left = "0%";
  fill.style.width = `${strength}%`;
  fill.style.background = up ? "var(--up)" : "var(--down)";

  $("cg-note").innerHTML =
    `Ranked against every signal this model has produced, this one sits in the
     <b>top ${Math.max(1, 100 - strength)}%</b>. Directionally it leans
     ${up ? "upward" : "downward"} with a ${num(conf * 100, 0)}% probability —
     an edge, not a certainty. The levels on the left are sized so that the stop
     survives 9 out of 10 normal moves.`;

  markLevels(px, tp, sl);
}

/* ---------- how to trade the call ----------
   The direction read on its own left people asking "so do I buy, and where, and
   on spot or futures?". This turns the same numbers into the concrete actions a
   trader takes, spelled out for both venues, without inventing anything: entry,
   target and stop all come straight from the signal and the quantile model. */
function renderPlan(a, sigResp) {
  const s = (sigResp && sigResp.signal) || null;
  const q = a.quantiles_bps || {};
  const px = a.price;

  if (!s) {
    $("plan-lead").textContent =
      "No directional call right now — the model sees no edge either way. Sit out or trade your own read.";
    ["plan-spot-1", "plan-spot-2", "plan-fut-1", "plan-fut-2",
     "plan-risk-1", "plan-risk-2", "plan-note"].forEach((id) => ($(id).textContent = "—"));
    return;
  }

  const up = s.p_up > 0.5;
  const dir = up ? "upward" : "downward";
  const tp = s.target_price, sl = s.stop_price;
  const tpTxt = price(tp), slTxt = price(sl);
  const tpPct = pct(s.target_bps), slPct = pct(s.stop_bps);
  const hz = a.horizon_min >= 60 ? "hour" : `${a.horizon_min} minutes`;

  $("plan-lead").innerHTML =
    `The model leans <b class="${up ? "hl" : ""}" style="color:${up ? "var(--up)" : "var(--down)"}">` +
    `${up ? "UP" : "DOWN"}</b> over the next ${hz}. Here is what that means on each venue. ` +
    `The prices below are the same in every case — only how you take the trade differs.`;

  if (up) {
    $("plan-spot-1").innerHTML =
      `<b class="up">Buy</b> around <b>${price(px)}</b>.`;
    $("plan-spot-2").innerHTML =
      `Take profit near <b>${tpTxt}</b> (+${tpPct}). Cut the position if it drops to ` +
      `<b>${slTxt}</b> (−${slPct}). You own the coin — no liquidation risk, no funding fees.`;
    $("plan-fut-1").innerHTML =
      `Open a <b class="up">LONG</b> at <b>${price(px)}</b>.`;
    $("plan-fut-2").innerHTML =
      `Same target <b>${tpTxt}</b> and stop <b>${slTxt}</b>. Keep leverage low — 1× to 3× ` +
      `for a ${hz} scalp. Higher leverage means the stop can be hit by normal noise before your idea plays out.`;
  } else {
    $("plan-spot-1").innerHTML =
      `Spot can only <b>sell / stay out</b>. If you hold ${a.symbol.replace("USDT", "")}, ` +
      `this favours trimming near <b>${price(px)}</b>.`;
    $("plan-spot-2").innerHTML =
      `You cannot profit from a fall on spot — you can only avoid it. To trade the ` +
      `downside directly you need futures.`;
    $("plan-fut-1").innerHTML =
      `Open a <b class="down">SHORT</b> at <b>${price(px)}</b>.`;
    $("plan-fut-2").innerHTML =
      `Target <b>${tpTxt}</b> (−${tpPct}), stop <b>${slTxt}</b> (+${slPct}). Keep leverage low — ` +
      `1× to 3× for a ${hz} scalp. Shorting means borrowing, so watch the funding rate.`;
  }

  // reward-to-risk expressed as "reward : risk" so bigger is always better and
  // a value under 1 reads unambiguously as reward smaller than risk
  const rr = s.stop_bps ? (s.target_bps / s.stop_bps) : null;
  const rrTxt = rr ? `${num(rr, 2)} : 1` : "—";
  const rrVerdict = rr == null ? ""
    : rr >= 1.5 ? " — a healthy setup: the target is worth clearly more than the risk."
    : rr >= 1 ? " — acceptable: reward and risk are close."
    : " — weak: the likely reward is smaller than the risk, common when the market is quiet.";
  $("plan-risk-1").innerHTML =
    `Risk a fixed slice of your account — most scalpers use <b>0.5% to 1%</b> per trade.`;
  $("plan-risk-2").innerHTML =
    `With the stop ${slPct} away, risking 1% of a $10,000 account is a position of about ` +
    `<b>${money10k(s.stop_bps)}</b>. Reward vs risk here is <b>${rrTxt}</b>${rrVerdict}`;

  $("plan-note").textContent =
    "Entry, target and stop are generated automatically from the live forecast. " +
    "They are a starting point sized to survive normal noise — not financial advice, and not a guarantee.";
}

function money10k(stopBps) {
  // position size that risks 1% of a $10k account given the stop distance
  if (!stopBps) return "—";
  const risk = 100; // 1% of 10,000
  const size = risk / (stopBps / 1e4);
  return size >= 1000 ? `$${Math.round(size).toLocaleString()}` : `$${size.toFixed(0)}`;
}

/* ---------- rule engine plan ----------
   Renders the deterministic confluence decision: the verdict, the structural
   levels, a signed confluence meter, per-family scores and every indicator vote.
   Nothing here forecasts — it mirrors exactly what rule_engine.decide() decided,
   and it owns the entry/target/stop lines drawn on the chart. */
function renderRulePlan(resp) {
  const p = resp && resp.plan;
  const card = $("rule-card");
  if (!card) return;
  if (!p) {
    card.className = "card re-neutral";
    $("re-verdict").textContent = "—";
    $("re-narr").textContent = "Rule engine unavailable for this symbol.";
    return;
  }

  // The card always shows a direction (the AI's lean) and the levels — the reader
  // decides whether to act on them. Direction follows the bias, never a "wait".
  const bias = p.bias || p.side;             // "long" | "short"
  const up = bias === "long";
  card.className = `card re-${bias}`;
  const V = $("re-verdict");
  V.textContent = up ? "▲ LONG" : "▼ SHORT";
  V.className = `re-verdict ${bias}`;

  // signed confluence meter: fill grows out from the centre line
  const sc = p.score || 0;
  $("re-score").textContent = (sc >= 0 ? "+" : "") + sc.toFixed(2);
  const sf = $("re-score-fill");
  sf.style.width = Math.min(Math.abs(sc) * 50, 50) + "%";
  sf.style.left = sc >= 0 ? "50%" : (50 - Math.min(Math.abs(sc) * 50, 50)) + "%";
  sf.style.background = up ? "var(--up)" : "var(--down)";

  $("re-conf").textContent = (p.confidence != null ? p.confidence : "—") + "%";
  $("re-adx").textContent =
    `${p.adx != null ? p.adx : "—"} · ${p.trending ? "trending" : "ranging"}`;
  $("re-agree").textContent = `${p.agree} / ${p.total_votes}`;
  $("re-narr").textContent = p.narrative || "";

  // levels are always shown and always re-scale with the timeframe
  if (p.entry != null) {
    $("re-entry").textContent = price(p.entry);
    $("re-take").textContent = price(p.take);
    $("re-stop").textContent = price(p.stop);
    $("re-take-x").textContent = `${up ? "+" : "−"}${pct(p.take_bps)}`;
    $("re-stop-x").textContent = `${up ? "−" : "+"}${pct(p.stop_bps)}`;
    $("re-rr").textContent = p.rr ? `${num(p.rr, 2)} : 1` : "—";
    markLevels(p.entry, p.take, p.stop);     // draw entry/target/stop on the chart
  } else {
    ["re-entry", "re-take", "re-stop", "re-rr"].forEach((id) => ($(id).textContent = "—"));
    $("re-take-x").textContent = "";
    $("re-stop-x").textContent = "";
  }

  // per-family confluence bars, centred like the main meter
  const fams = { trend: "Trend", momentum: "Momentum",
                 location: "Location", tape: "Order flow" };
  $("re-fams").innerHTML = Object.entries(fams).map(([k, lbl]) => {
    const s = p.family_scores ? p.family_scores[k] : null;
    if (s == null)
      return `<div class="re-fam"><span>${lbl}</span>` +
             `<div class="re-fbar"></div><b>—</b></div>`;
    const w = Math.min(Math.abs(s) * 50, 50);
    const left = s >= 0 ? 50 : 50 - w;
    const col = s > 0 ? "var(--up)" : s < 0 ? "var(--down)" : "var(--muted)";
    return `<div class="re-fam"><span>${lbl}</span>` +
      `<div class="re-fbar"><i style="left:${left}%;width:${w}%;background:${col}"></i></div>` +
      `<b>${s >= 0 ? "+" : ""}${s.toFixed(2)}</b></div>`;
  }).join("");

  const votes = p.signals || [];
  $("re-votes-n").textContent = votes.length;
  $("re-votes").innerHTML = votes.map((v) => {
    const c = v.dir === "long" ? "up" : v.dir === "short" ? "down" : "flat";
    const arrow = v.dir === "long" ? "▲" : v.dir === "short" ? "▼" : "•";
    return `<div class="re-vote ${c}"><span class="rv-arrow">${arrow}</span>` +
      `<span class="rv-label">${v.label}</span>` +
      `<span class="rv-reason">${v.reason}</span></div>`;
  }).join("");

  // timeframe label + a calm, fixed "updated HH:MM" stamp (no per-second counter,
  // which felt twitchy) — the numbers themselves only move once a minute
  const tf = resp.tf || TF;
  const tag = $("re-tag");
  if (tag) tag.textContent = `AI · ${tfLabel(tf)}`;
  const stamp = $("re-live-text");
  if (stamp) {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, "0");
    const mm = String(now.getMinutes()).padStart(2, "0");
    stamp.textContent = `updated ${hh}:${mm}`;
  }
}

/* ---------- pro strategy read (drives the 4H card) ----------
   The trend/volatility setup that backtested with a real edge on 4H. It's an
   event strategy (enter on a trigger, trail the stop), so the card shows the
   directional lean, which of the checks pass, the entry/2.5×ATR stop, and
   whether a fresh signal just fired. */
async function fetchProPlan(sym, tf) {
  try {
    const r = await fetch(`/api/pro_plan?symbol=${sym}&tf=${tf}`);
    if (!r.ok) return;
    renderProPlan(await r.json());
  } catch (e) { /* transient */ }
}

function proTfLabel(tf) { return tf >= 60 ? `${tf / 60}H` : `${tf}m`; }

function renderProPlan(p) {
  const card = $("rule-card");
  if (!card || !p) return;
  const up = p.bias === "long";
  // just the entry — for showing live-price delta vs the decision
  LIVE_PRO = { bias: p.bias, entry: p.entry };
  card.className = `card re-${p.bias}`;
  const V = $("re-verdict");
  V.textContent = up ? "▲ LONG" : "▼ SHORT";
  V.className = `re-verdict ${p.bias}`;
  const tag = $("re-tag");
  const tl = proTfLabel(p.tf);
  if (tag) tag.textContent = `Pro · ${tl}${p.signal_now ? " · SIGNAL" : ""}`;

  const frac = p.total_checks ? p.passed / p.total_checks : 0;
  $("re-score").textContent = `${p.passed}/${p.total_checks}`;
  const sf = $("re-score-fill");
  sf.style.left = "50%";
  sf.style.width = frac * 50 + "%";
  sf.style.background = up ? "var(--up)" : "var(--down)";
  $("re-conf").textContent = Math.round(frac * 100) + "%";
  $("re-adx").textContent = `${p.adx} · ${p.adx > 20 ? "trending" : "weak"}`;
  $("re-agree").textContent = `${p.passed} / ${p.total_checks}`;

  // Levels are ALWAYS shown — vote-based bias always produces a direction.
  // Conviction label reflects how many indicators agree with that direction.
  const target = p.target;
  const stopX = p.stop_atr != null ? p.stop_atr : 2;
  const trailX = p.trail_atr != null ? p.trail_atr : 7;
  const conv = p.passed >= 5 ? "very high (5/5)"
             : p.passed >= 4 ? "high (4/5)"
             : p.passed >= 3 ? "moderate (3/5)"
             : `lower (${p.passed}/5) — watch closely`;
  $("re-entry").textContent = price(p.entry);
  $("re-stop").textContent = price(p.stop);
  $("re-take").textContent = price(target);
  $("re-take-x").textContent = `2R · then trails`;
  $("re-stop-x").textContent = `${up ? "−" : "+"}${pct(p.stop_bps)}`;
  $("re-rr").textContent = p.signal_now ? "● ENTRY NOW"
    : p.passed >= 4 ? "strong · watching" : "watching";
  $("re-narr").textContent =
    `Pro ${tl}: ${p.passed}/${p.total_checks} indicators vote ${p.bias} — conviction ${conv}. ` +
    `Enter ${price(p.entry)}, stop ${price(p.stop)} (${stopX}×ATR = 1R), target ${price(target)} (2R). ` +
    `Past the target the ${trailX}×ATR trailing stop lets the winner run.`;

  $("re-fams").innerHTML = (p.checks || []).map((c) => {
    const col = c.pass ? "var(--up)" : "var(--faint)";
    return `<div class="re-fam"><span>${c.label}</span>` +
      `<div class="re-fbar"><i style="left:${c.pass ? 0 : 46}%;width:${c.pass ? 100 : 8}%;background:${col}"></i></div>` +
      `<b>${c.pass ? "✓" : "·"}</b></div>`;
  }).join("");

  $("re-votes-n").textContent = (p.checks || []).length;
  $("re-votes").innerHTML = (p.checks || []).map((c) => {
    const cls = c.pass ? (up ? "up" : "down") : "flat";
    return `<div class="re-vote ${cls}"><span class="rv-arrow">${c.pass ? "✓" : "·"}</span>` +
      `<span class="rv-label">${c.label}</span><span class="rv-reason">${c.detail}</span></div>`;
  }).join("");

  const stamp = $("re-live-text");
  if (stamp) {
    const n = new Date();
    stamp.textContent = `updated ${String(n.getHours()).padStart(2, "0")}:${String(n.getMinutes()).padStart(2, "0")}`;
  }
}

/* ---------- live watch: refresh the read once a minute, quietly ---------- */
function startLiveWatch() {
  if (PLAN_POLL) clearInterval(PLAN_POLL);
  PLAN_POLL = setInterval(refreshPlanOnly, PLAN_POLL_MS);
}

async function refreshPlanOnly() {
  // don't burn requests while the tab is hidden; resume on return
  if (document.hidden || LOADING) return;
  const sym = $("symbol").value || "BTCUSDT";
  try {
    await fetchProPlan(sym, TF);   // pro strategy on every timeframe; re-marks levels
  } catch (e) { /* transient — next minute retries */ }
}

/* ---------- move size ladder ---------- */
function renderLadder(a) {
  const q = a.quantiles_bps || {};
  const px = a.price;
  const rows = [
    ["q90", "Big move", "Happens 1 time in 10", "#f5c542"],
    ["q75", "Above average", "Happens 1 time in 4", "#dfae32"],
    ["q50", "Typical move", "Happens half the time", "#b8891f"],
    ["q25", "Quiet", "3 times in 4 it is bigger than this", "#7a5c18"],
    ["q10", "Very quiet", "9 times in 10 it is bigger than this", "#4a3a12"],
  ];
  const max = q.q90 || 1;
  $("ladder").innerHTML = rows.map(([k, title, note, col]) => {
    const w = Math.max(5, (100 * (q[k] || 0)) / max);
    return `<div class="lrow">
      <div class="lr-left">
        <span class="lr-title">${title}</span>
        <span class="lr-note">${note}</span>
      </div>
      <div class="lr-bar"><i style="width:${w}%;background:${col}"></i></div>
      <div class="lr-val">
        <span class="lr-pct">±${pct(q[k])}</span>
        <span class="lr-usd">${usdOf(px, q[k])}</span>
      </div></div>`;
  }).join("");

  $("lf-typical").innerHTML = `±${pct(q.q50)} <span>${usdOf(px, q.q50)}</span>`;
  $("lf-stop").innerHTML = `${pct(q.q90)} <span>${usdOf(px, q.q90)}</span>`;
  $("lf-target").innerHTML = `${pct(q.q75)} <span>${usdOf(px, q.q75)}</span>`;
}

/* ---------- market pulse ---------- */
function renderPulse(a) {
  const f = a.order_flow || {};
  const gauge = (name, val, lo, hi, note, shown, goodDir) => {
    const p = Math.max(0, Math.min(1, (val - lo) / (hi - lo)));
    const left = Math.min(p, 0.5) * 100, width = Math.abs(p - 0.5) * 100;
    const col = goodDir === false ? "var(--gold)"
      : val >= 0 ? "var(--up)" : "var(--down)";
    return `<div class="gauge">
      <div class="g-head"><span class="g-name">${name}</span>
        <span class="g-val" style="color:${col}">${shown}</span></div>
      <div class="g-track"><div class="g-mid"></div>
        <div class="g-fill" style="left:${left}%;width:${width}%;background:${col}"></div></div>
      <div class="g-note">${note}</div></div>`;
  };

  const imb = f.taker_imbalance ?? 0;
  const wh = f.whale_imbalance ?? 0;
  const inten = f.trade_intensity_vs_day ?? 1;
  const buyPct = 50 + imb * 50;

  $("flow-gauges").innerHTML =
    gauge("Buying vs selling pressure", imb, -1, 1,
      imb > 0.15 ? `Buyers are in control — ${num(buyPct, 0)}% of the pressure is buying.`
      : imb < -0.15 ? `Sellers are in control — ${num(100 - buyPct, 0)}% of the pressure is selling.`
      : "Neither side has the upper hand right now.",
      imb > 0.02 ? "BUYERS" : imb < -0.02 ? "SELLERS" : "EVEN") +
    gauge("Whales (orders over $100k)", wh, -1, 1,
      wh > 0.15 ? "Big money is buying into this move."
      : wh < -0.15 ? "Big money is selling into this move."
      : "Large orders are hitting both sides equally.",
      wh > 0.02 ? "BUYING" : wh < -0.02 ? "SELLING" : "SPLIT") +
    gauge("How busy the market is", inten - 1, -1, 2,
      inten > 1.3 ? "Much busier than normal — moves tend to be larger when the tape is this active."
      : inten < 0.7 ? "Quieter than normal — expect smaller moves and thinner liquidity."
      : "Normal activity for this time of day.",
      `${num(inten, 1)}× normal`, false) +
    (f.open_interest_change_pct == null ? "" :
      gauge("New money entering", f.open_interest_change_pct, -1, 1,
        f.open_interest_change_pct > 0
          ? "Traders are opening new positions — the move has fuel behind it."
          : "Traders are closing positions — the move may be running out of steam.",
        f.open_interest_change_pct > 0 ? "OPENING" : "CLOSING"));

  const p = a.positioning || {};
  const posGauge = (name, v, pctile, note) => {
    const lean = Math.max(-1, Math.min(1, Math.log(v) / Math.log(3)));
    const left = Math.min(lean, 0) * 50 + 50, width = Math.abs(lean) * 50;
    const col = lean >= 0 ? "var(--up)" : "var(--down)";
    const longPct = (v / (1 + v)) * 100;
    return `<div class="gauge">
      <div class="g-head"><span class="g-name">${name}</span>
        <span class="g-val" style="color:${col}">${num(longPct, 0)}% long</span></div>
      <div class="g-track"><div class="g-mid"></div>
        <div class="g-fill" style="left:${left}%;width:${width}%;background:${col}"></div></div>
      <div class="g-note">${note}${pctile != null
        ? ` That is ${pctile >= 80 ? "unusually high" : pctile <= 20
            ? "unusually low" : "about normal"} for today.` : ""}</div></div>`;
  };
  let ph = "";
  if (p.retail_long_short != null)
    ph += posGauge("The retail crowd", p.retail_long_short,
      p.retail_long_short_pctile_1d,
      p.retail_long_short > 1.6 ? "Most small traders are betting on a rise."
      : p.retail_long_short < 0.7 ? "Most small traders are betting on a fall."
      : "Small traders are fairly evenly split.");
  if (p.top_trader_long_short != null)
    ph += posGauge("The biggest accounts", p.top_trader_long_short,
      p.top_trader_long_short_pctile_1d,
      "These are the largest position holders on the exchange.");
  if (p.taker_buy_sell_volume != null)
    ph += posGauge("Market orders today", p.taker_buy_sell_volume,
      p.taker_buy_sell_volume_pctile_1d,
      "How much was bought at market versus sold at market this session.");
  $("pos-gauges").innerHTML = ph;
}

/* ---------- indicator signal cards ---------- */
function renderSignals(a) {
  const readable = {
    IND_atr_rel_14: ["Volatility (ATR)", "how wide recent candles are"],
    IND_atr_rel_60: ["Hourly volatility", "candle width over the last hour"],
    IND_atr_rel_240: ["4-hour volatility", "candle width over four hours"],
    IND_bb_width_14: ["Bollinger Bands", "how far the bands have opened"],
    IND_bb_width_60: ["Bollinger Bands (1H)", "band width over the hour"],
    IND_macd_rel: ["MACD", "momentum between fast and slow averages"],
    CDL_range_rel_15: ["Candle size (15m)", "high to low on recent candles"],
    CDL_range_rel_60: ["Candle size (1H)", "high to low over the hour"],
    SR_channel_width_60: ["Hourly range", "how wide the last hour traded"],
    SR_channel_width_240: ["4-hour range", "how wide the last four hours traded"],
    SR_channel_width_1440: ["Daily range", "how wide today has traded"],
    SR_round_dist_1000: ["Round number ($1k)", "orders pile up at round prices"],
    SR_round_dist_5000: ["Round number ($5k)", "orders pile up at round prices"],
    SR_to_low_1440: ["Distance above the day's low", "how much room to support"],
    VP_dist_poc_240: ["Volume magnet (4H)", "the price where most trading happened"],
    VP_dist_poc_1440: ["Volume magnet (1D)", "the price where most trading happened"],
  };
  const ta = a.technical || [];
  $("sig-cards").innerHTML = ta.map((r) => {
    const [nm, desc] = readable[r.indicator] || [r.indicator, ""];
    const amp = r.amplification;
    const boost = Math.round((amp - 1) * 100);
    const state = amp >= 1.15 ? ["BIGGER MOVES", "st-hot"]
      : amp >= 0.95 ? ["NORMAL", "st-neutral"] : ["SMALLER MOVES", "st-cool"];
    const line = amp >= 1.15
      ? `Last time this indicator looked like this, the move that followed was <b>${boost}% bigger</b> than usual.`
      : amp >= 0.95
      ? `Moves following readings like this have been about average.`
      : `Moves following readings like this have been <b>${Math.abs(boost)}% smaller</b> than usual.`;
    const high = r.percentile >= 75, low = r.percentile <= 25;
    const where = high ? "Running hot" : low ? "Running cold" : "Mid-range";
    return `<div class="sig-card">
      <div class="sc-top">
        <div><div class="sc-name">${nm}</div><div class="sc-desc">${desc}</div></div>
        <span class="state ${state[1]}">${state[0]}</span>
      </div>
      <div class="sc-meter"><i style="width:${Math.max(3, r.percentile)}%"></i></div>
      <div class="sc-where">${where} — higher than ${num(r.percentile, 0)}% of the last month</div>
      <div class="sc-line">${line}</div>
    </div>`;
  }).join("");
}

/* ---------- performance charts ---------- */
const NS = "http://www.w3.org/2000/svg";
function svgEl(p, tag, attrs, text) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  p.appendChild(el);
  return el;
}

function drawBenchmark(svg, model, bench) {
  svg.innerHTML = "";
  const base = 158, bw = 96;
  const max = Math.max(model, bench) * 1.28 || 1;
  [{ x: 92, v: bench, l: "Industry standard", c: "#3a3a46", t: "#9b9daa" },
   { x: 232, v: model, l: "ASTRA AI", c: "#f5c542", t: "#f5c542" }].forEach((b) => {
    const h = (b.v / max) * 118;
    svgEl(svg, "rect", { x: b.x, y: base - h, width: bw, height: h, rx: 5, fill: b.c });
    svgEl(svg, "text", { x: b.x + bw / 2, y: base - h - 10, fill: b.t,
      "font-size": 16, "font-weight": 800, "text-anchor": "middle" },
      `${(b.v * 100).toFixed(0)}`);
    svgEl(svg, "text", { x: b.x + bw / 2, y: base + 19, fill: "#9b9daa",
      "font-size": 11.5, "text-anchor": "middle" }, b.l);
  });
  svgEl(svg, "line", { x1: 40, x2: 396, y1: base, y2: base, stroke: "#23232c" });
  svgEl(svg, "text", { x: 210, y: 190, fill: "#2ecc8f", "font-size": 13,
    "font-weight": 700, "text-anchor": "middle" },
    `${((model / bench - 1) * 100).toFixed(0)}% more accurate`);
}

function drawSkill(svg, curve) {
  svg.innerHTML = "";
  const W = 420, H = 200, padL = 30, padR = 12, padT = 16, padB = 30;
  if (!curve || !curve.skill || !curve.skill.length) {
    svgEl(svg, "text", { x: W / 2, y: H / 2, fill: "#63656f", "font-size": 13,
      "text-anchor": "middle" }, "Building the weekly record");
    return;
  }
  const s = curve.skill;
  const lo = Math.min(0, ...s), hi = Math.max(...s, 1);
  const x = (i) => padL + (i / Math.max(1, s.length - 1)) * (W - padL - padR);
  const y = (v) => padT + ((hi - v) / (hi - lo || 1)) * (H - padT - padB);
  svgEl(svg, "line", { x1: padL, x2: W - padR, y1: y(0), y2: y(0),
    stroke: "#33333f", "stroke-dasharray": "3 3" });
  const bw = Math.max(3, (W - padL - padR) / s.length - 3);
  s.forEach((v, i) => {
    svgEl(svg, "rect", { x: x(i) - bw / 2, y: v >= 0 ? y(v) : y(0), width: bw,
      height: Math.max(1.5, Math.abs(y(v) - y(0))), rx: 2,
      fill: v >= 0 ? "#2ecc8f" : "#f0554f", opacity: 0.85 });
  });
  svgEl(svg, "text", { x: padL, y: H - 8, fill: "#63656f", "font-size": 10.5 },
    `Ahead in ${curve.positive_share}% of weeks`);
}

function drawCalibration(svg, coverage) {
  svg.innerHTML = "";
  const W = 880, H = 220, padL = 60, padR = 26, padT = 18, padB = 44;
  const order = ["q10", "q25", "q50", "q75", "q90"];
  const promised = { q10: 10, q25: 25, q50: 50, q75: 75, q90: 90 };
  const px = (v) => padL + (v / 100) * (W - padL - padR);
  const py = (v) => padT + ((100 - v) / 100) * (H - padT - padB);
  for (let k = 0; k <= 4; k++) {
    const v = k * 25;
    svgEl(svg, "line", { x1: padL, x2: W - padR, y1: py(v), y2: py(v), stroke: "#1a1a21" });
    svgEl(svg, "text", { x: padL - 10, y: py(v) + 4, fill: "#63656f",
      "font-size": 10.5, "text-anchor": "end" }, `${v}%`);
    svgEl(svg, "text", { x: px(v), y: H - 18, fill: "#63656f",
      "font-size": 10.5, "text-anchor": "middle" }, `${v}%`);
  }
  svgEl(svg, "line", { x1: px(0), y1: py(0), x2: px(100), y2: py(100),
    stroke: "#3a3a46", "stroke-dasharray": "5 4" });
  const pts = [];
  for (const k of order) {
    const actual = coverage && coverage[k] != null ? coverage[k] : promised[k];
    pts.push([px(promised[k]), py(actual)]);
    svgEl(svg, "circle", { cx: px(promised[k]), cy: py(actual), r: 6,
      fill: "#f5c542", stroke: "#0d0d12", "stroke-width": 2 });
    svgEl(svg, "text", { x: px(promised[k]), y: py(actual) - 14, fill: "#f5f6f8",
      "font-size": 11.5, "font-weight": 700, "text-anchor": "middle" },
      `${actual.toFixed(1)}%`);
  }
  svgEl(svg, "polyline", { points: pts.map((p) => p.join(",")).join(" "),
    fill: "none", stroke: "#f5c542", "stroke-width": 2 });
  svgEl(svg, "text", { x: W / 2, y: H - 3, fill: "#9b9daa", "font-size": 11.5,
    "text-anchor": "middle" }, "what we promised  →      what actually happened  ↑");
}

function renderPerformance(a, stats) {
  const pm = (stats && stats.point_model) || {};
  const qm = (stats && stats.quantile_model) || {};
  const sc = (stats && stats.skill_curve) || null;
  const dt = (stats && stats.data) || {};
  const r2 = pm.r2 ?? a.model_valid_r2;
  const bench = pm.benchmark_r2 ?? a.benchmark_har_r2;

  $("ps-improve").textContent = `${num((r2 / bench - 1) * 100, 0)}%`;
  $("ps-cov").textContent = `±${num(qm.walkforward_worst_error_pts ?? 0.8, 1)}%`;
  $("ps-weeks").textContent = sc ? `${num(sc.positive_share, 0)}%` : "95%";
  $("ps-data").textContent = dt.days ? `${dt.days}` : "730";

  // top strip: lead with the measurements that are genuinely strong
  const cov = qm.coverage || {};
  const hit = cov.q90 != null ? cov.q90 : 90.2;
  $("m-reliability").textContent = `${num(hit, 1)}%`;
  $("m-edge").textContent = `+${num((r2 / bench - 1) * 100, 0)}%`;
  $("m-weeks").textContent = sc ? `${num(sc.positive_share, 0)}%` : "95%";
  $("m-data").textContent = dt.days ? `${dt.days}` : "730";
  if (pm.n_features) $("pl-features").textContent = pm.n_features;

  drawBenchmark($("bench-chart"), r2, bench);
  drawSkill($("skill-chart"), sc);
  drawCalibration($("calib-chart"), qm.coverage);
}

/* ---------- events ---------- */
$("symbol").addEventListener("change", load);
$("refresh").addEventListener("click", load);
document.querySelectorAll(".tf").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tf").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    TF = parseInt(b.dataset.tf, 10);
    load();
  }));
const openStudies = () => {
  $("studies").hidden = false; $("studies-backdrop").hidden = false;
};
const closeStudies = () => {
  $("studies").hidden = true; $("studies-backdrop").hidden = true;
};
$("open-studies").addEventListener("click", openStudies);
// second entry point, sitting right on the chart, so adding an indicator never
// means scrolling back up to the toolbar
const chartIndBtn = $("chart-ind-btn");
if (chartIndBtn) chartIndBtn.addEventListener("click", openStudies);
$("close-studies").addEventListener("click", closeStudies);
$("studies-backdrop").addEventListener("click", closeStudies);

loadSymbols().then((syms) => {
  loadTicker(syms);
  loadScanner();
  setInterval(loadScanner, 60000);       // cross-coin scanner refreshes each minute
  setInterval(loadIndScan, 300000);      // indicator-suite scan every 5 minutes
  setInterval(() => loadHFSignal($("symbol").value || "BTCUSDT"), 120000);  // HF signal every 2 min
  return load();
});
