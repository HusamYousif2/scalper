/* report.js — the performance page. Fetches the backtest scorecard, the weekly
   history, and the trade list, and draws a small equity curve. No dependencies. */

const $ = (id) => document.getElementById(id);
let TF = 240;                   // the strategy's edge is on 4H — the report is locked to it
let SEQ = 0;
let LAST_REP = null, LAST_WEEKS = [];

function fmtPx(v) {
  if (v == null) return "—";
  const a = Math.abs(v);
  const dp = a >= 1000 ? 1 : a >= 1 ? 2 : 5;
  return v.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
}
function fmtR(v) { return (v >= 0 ? "+" : "") + Number(v).toFixed(2) + "R"; }
function dt(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function dOnly(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

async function loadSymbols() {
  try {
    const j = await (await fetch("/api/symbols?data=1")).json();
    const sel = $("rp-symbol");
    sel.innerHTML = "";
    for (const s of j.symbols) {
      const o = document.createElement("option");
      o.value = s.symbol;
      o.textContent = s.symbol.replace("USDT", " / USDT");
      sel.appendChild(o);
    }
    if ([...sel.options].some((o) => o.value === "BTCUSDT")) sel.value = "BTCUSDT";
  } catch (e) { /* leave empty */ }
}

async function load() {
  const seq = ++SEQ;
  const sym = $("rp-symbol").value || "BTCUSDT";
  const days = parseInt($("rp-days").value, 10) || 90;
  $("rp-status").hidden = false;
  $("rp-status").textContent = `Scoring the engine on ${sym.replace("USDT", "/USDT")} · ${tfLabel(TF)} over ${days} days…`;
  $("rp-body").hidden = true;

  try {
    const [rep, weeksResp] = await Promise.all([
      fetch(`/api/backtest?symbol=${sym}&tf=${TF}&days=${days}`)
        .then((r) => { if (!r.ok) throw new Error(r.statusText); return r.json(); }),
      fetch(`/api/perf/weeks?symbol=${sym}&tf=${TF}`)
        .then((r) => (r.ok ? r.json() : { weeks: [] })).catch(() => ({ weeks: [] })),
    ]);
    if (seq !== SEQ) return;
    LAST_REP = rep; LAST_WEEKS = weeksResp.weeks || [];
    render(rep, LAST_WEEKS);
    $("rp-status").hidden = true;
    $("rp-body").hidden = false;
  } catch (e) {
    if (seq !== SEQ) return;
    $("rp-status").textContent = `Could not score: ${e.message}. The archive may still be warming up — try again shortly.`;
  }
}

function tfLabel(tf) { return tf >= 60 ? `${tf / 60}H` : `${tf}m`; }

function render(r, weeks) {
  const b = r.gross || r.net || r;
  const profitable = b.total_r > 0;

  $("rp-title").textContent = `${r.symbol.replace("USDT", " / USDT")} · ${tfLabel(r.tf)}`;
  const wb = $("rp-winbadge");
  wb.textContent = `${fmtR(b.total_r)} · ${profitable ? "profitable" : "losing"}`;
  wb.className = `win-badge ${profitable ? "good" : "bad"}`;
  $("rp-windowtxt").textContent = r.from && r.to ? `${dOnly(r.from)} → ${dOnly(r.to)}` : "";

  $("k-trades").textContent = b.trades;
  $("k-trades-sub").textContent = `${b.wins} won · ${b.losses} lost`;
  $("k-winrate").textContent = `${b.win_rate}%`;
  $("k-winrate-sub").textContent = `${b.wins} of ${b.trades} hit target`;
  const net = $("k-net");
  net.textContent = fmtR(b.total_r);
  net.className = `t-val ${b.total_r >= 0 ? "up" : "down"}`;
  $("k-pf").textContent = b.profit_factor == null ? "∞" : b.profit_factor;
  $("k-wins").textContent = b.wins;
  $("k-losses").textContent = b.losses;
  const exp = $("k-exp");
  exp.textContent = fmtR(b.expectancy_r);
  exp.className = `t-val ${b.expectancy_r >= 0 ? "up" : "down"}`;
  $("k-dd").textContent = `${Number(b.max_drawdown_r).toFixed(2)}R`;

  drawEquity($("rp-equity"), b.equity || []);
  renderWeeks(weeks);
  renderTrades(r.trades_list || []);
}

async function loadPortfolio() {
  try {
    const p = await fetch("/api/portfolio?tf=240&days=190").then((r) => {
      if (!r.ok) throw new Error(r.statusText); return r.json();
    });
    renderPortfolio(p);
  } catch (e) {
    $("pf-badge").textContent = "unavailable";
  }
}

// compound a cumulative-R equity curve into account growth at a fixed risk/trade
function compound(equity, riskPct) {
  const risk = riskPct / 100;
  let e = 1, peak = 1, dd = 0, prev = 0;
  const curve = [{ r: 0 }];
  for (const pt of equity) {
    const dr = pt.r - prev; prev = pt.r;
    e *= (1 + dr * risk);
    peak = Math.max(peak, e);
    dd = Math.min(dd, e / peak - 1);
    curve.push({ time: pt.time, r: (e - 1) * 100 });   // reuse drawEquity (plots .r)
  }
  return { growthPct: (e - 1) * 100, maxDdPct: dd * 100, final: e, curve };
}

function renderPortfolio(p) {
  const m = p.metrics || {};
  const good = m.net > 0;
  const RISK = 1;   // percent of account risked per trade
  const eq = p.equity || [];
  const c = compound(eq, RISK);
  const days = p.from && p.to ? Math.max(1, (p.to - p.from) / 86400) : 0;
  const annual = days ? (Math.pow(c.final, 365 / days) - 1) * 100 : 0;

  const wb = $("pf-badge");
  wb.textContent = `${good ? "+" : ""}${c.growthPct.toFixed(1)}% · ${good ? "profitable" : "losing"}`;
  wb.className = `win-badge ${good ? "good" : "bad"}`;
  $("pf-window").textContent = p.from && p.to ? `${dOnly(p.from)} → ${dOnly(p.to)} · 4H` : "4H";

  const gr = $("pf-growth");
  gr.textContent = `${c.growthPct >= 0 ? "+" : ""}${c.growthPct.toFixed(1)}%`;
  gr.className = `t-val ${c.growthPct >= 0 ? "up" : "down"}`;
  const an = $("pf-annual");
  an.textContent = `${annual >= 0 ? "+" : ""}${annual.toFixed(0)}%`;
  an.className = `t-val ${annual >= 0 ? "up" : "down"}`;
  const net = $("pf-net");
  net.textContent = fmtR(m.net);
  net.className = `t-val ${good ? "up" : "down"}`;
  $("pf-trades-sub").textContent = `${m.n} trades · ${(p.per_symbol || []).length} symbols`;
  $("pf-win").textContent = `${m.win}%`;
  $("pf-pf").textContent = m.pf == null ? "∞" : m.pf;
  $("pf-dd").textContent = `${c.maxDdPct.toFixed(1)}%`;

  drawEquity($("pf-equity"), c.curve.slice(1));

  $("pf-persym").innerHTML = (p.per_symbol || []).map((s) => {
    const pos = s.net >= 0;
    return `<tr>
      <td>${s.symbol.replace("USDT", "")}</td>
      <td>${s.n}</td>
      <td>${s.win}%</td>
      <td class="${pos ? "win" : "loss"}">${fmtR(s.net)}</td>
      <td class="win">${s.avgW}</td>
      <td class="loss">${s.avgL}</td>
      <td>${s.pf == null ? "∞" : s.pf}</td>
    </tr>`;
  }).join("");
}

async function loadSignals() {
  try {
    const j = await fetch("/api/signals?tf=240").then((r) => r.json());
    renderSignals(j.rows || []);
  } catch (e) { /* leave last render */ }
}

function ratingColor(r) {
  return r === "Fire" ? "var(--gold)" : r === "Strong" ? "var(--up)"
    : r === "Building" ? "#5b9bf5" : "var(--faint)";
}
function scoreBar(score, rating) {
  const col = ratingColor(rating);
  return `<span class="sig-bar"><i style="width:${score}%;background:${col}"></i></span>` +
    `<b style="color:${col}">${score}</b>`;
}
function biasTag(b) { return `<span class="side-${b}">${b === "long" ? "▲ LONG" : "▼ SHORT"}</span>`; }

function renderSignals(rows) {
  const tb = $("sig-rows");
  if (!tb) return;
  if (!rows.length) {
    tb.innerHTML = `<tr><td colspan="8" class="rp-empty">Scanning…</td></tr>`;
    return;
  }
  const fires = rows.filter((r) => r.signal_now).length;
  const strong = rows.filter((r) => r.score >= 72).length;
  const badge = $("sig-badge");
  if (badge) {
    badge.textContent = fires ? `${fires} firing now` : strong ? `${strong} strong` : "watching";
    badge.className = `win-badge ${fires ? "good" : ""}`;
  }
  const upd = $("sig-updated");
  if (upd) {
    const n = new Date();
    upd.textContent = `4H · updated ${String(n.getHours()).padStart(2, "0")}:${String(n.getMinutes()).padStart(2, "0")}`;
  }

  // featured top pick
  const t = rows[0];
  const top = $("sig-top");
  if (top && t) {
    const col = ratingColor(t.rating);
    top.innerHTML = `<div class="sig-top ${t.bias}">
      <div class="sig-top-l">
        <div class="sig-top-coin">${t.symbol.replace("USDT", "")}<span> / USDT</span></div>
        <div class="sig-top-rating" style="color:${col}">${t.rating === "Fire" ? "● FIRE — entry triggered" : t.rating + " setup"} · ${biasTag(t.bias)}</div>
      </div>
      <div class="sig-top-score"><div class="sig-top-num" style="color:${col}">${t.score}</div><span>opportunity</span></div>
      <div class="sig-top-lvls">
        <div><span>Entry</span><b>${fmtPx(t.entry)}</b></div>
        <div><span>Stop</span><b class="loss">${fmtPx(t.stop)}</b></div>
        <div><span>Target</span><b class="win">${fmtPx(t.target)}</b></div>
      </div>
    </div>`;
  }

  tb.innerHTML = rows.map((r, i) => {
    const live = r.signal_now;
    return `<tr${live ? ' style="background:rgba(245,197,66,0.06)"' : ""}>
      <td>${i + 1}</td>
      <td><b>${r.symbol.replace("USDT", "")}</b></td>
      <td class="sig-score">${scoreBar(r.score, r.rating)}</td>
      <td>${biasTag(r.bias)}</td>
      <td style="color:${ratingColor(r.rating)}">${live ? "● FIRE" : r.rating}</td>
      <td>${fmtPx(r.entry)}</td>
      <td class="loss">${fmtPx(r.stop)}</td>
      <td class="win">${fmtPx(r.target)}</td>
    </tr>`;
  }).join("");
}

async function loadForward() {
  try {
    const f = await fetch("/api/forward?tf=240").then((r) => {
      if (!r.ok) throw new Error(r.statusText); return r.json();
    });
    renderForward(f);
  } catch (e) {
    $("fw-badge").textContent = "unavailable";
  }
}

function renderForward(f) {
  const m = f.metrics || {};
  const n = m.trades || 0;
  const good = m.total_r > 0;
  const wb = $("fw-badge");
  if (!n) { wb.textContent = "tracking · no trades yet"; wb.className = "win-badge"; }
  else { wb.textContent = `${fmtR(m.total_r)} · ${good ? "net positive" : "net negative"}`;
         wb.className = `win-badge ${good ? "good" : "bad"}`; }
  $("fw-window").textContent = f.start_ts ? `live since ${dOnly(f.start_ts)} · 4H` : "4H";

  const net = $("fw-net");
  net.textContent = n ? fmtR(m.total_r) : "—";
  net.className = `t-val ${good ? "up" : "down"}`;
  $("fw-trades").textContent = n;
  $("fw-since").textContent = f.start_ts ? dOnly(f.start_ts) : "—";
  $("fw-win").textContent = n ? `${m.win_rate}%` : "—";
  $("fw-pf").textContent = m.profit_factor == null ? "∞" : m.profit_factor;
  $("fw-dd").textContent = `${Number(m.max_drawdown_r).toFixed(1)}R`;

  drawEquity($("fw-equity"), m.equity || []);

  const tb = $("fw-tradelist");
  const list = f.trades_list || [];
  if (!list.length) {
    tb.innerHTML = `<tr><td colspan="9" class="rp-empty">No settled trades yet. The
      forward record begins ${f.start_ts ? dOnly(f.start_ts) : "now"} — every trade the
      strategy fires from here is frozen in, out-of-sample.</td></tr>`;
    return;
  }
  tb.innerHTML = list.map((t) => {
    const win = t.outcome === "win";
    return `<tr>
      <td>${dt(t.entry_time)}</td>
      <td>${(t.symbol || "").replace("USDT", "")}</td>
      <td class="side-${t.side}">${t.side === "long" ? "▲ LONG" : "▼ SHORT"}</td>
      <td>${fmtPx(t.entry)}</td>
      <td>${fmtPx(t.sl)}</td>
      <td>${fmtPx(t.tp)}</td>
      <td>${fmtPx(t.exit)}</td>
      <td class="${win ? "win" : "loss"}">${fmtR(t.r)}</td>
      <td class="${win ? "win" : "loss"}">${win ? "WIN" : "LOSS"}</td>
    </tr>`;
  }).join("");
}

function drawEquity(svg, eq) {
  svg.innerHTML = "";
  const W = svg.clientWidth || 1000, H = 220, pad = 10;
  if (!eq.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" fill="#63656f" font-size="13" text-anchor="middle">No trades in this window</text>`;
    return;
  }
  // series starts at 0, then each trade's cumulative R
  const ys = [0, ...eq.map((e) => e.r)];
  const min = Math.min(...ys), max = Math.max(...ys);
  const range = (max - min) || 1;
  const x = (i) => pad + (i / (ys.length - 1)) * (W - 2 * pad);
  const y = (v) => H - pad - ((v - min) / range) * (H - 2 * pad);

  const zeroY = y(0);
  const up = ys[ys.length - 1] >= 0;
  const col = up ? "#2ecc8f" : "#f0554f";
  const pts = ys.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const area = `${pad},${zeroY} ${pts} ${(W - pad).toFixed(1)},${zeroY}`;

  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML =
    `<line x1="${pad}" y1="${zeroY.toFixed(1)}" x2="${W - pad}" y2="${zeroY.toFixed(1)}" stroke="#2a2e39" stroke-width="1"/>` +
    `<polygon points="${area}" fill="${col}" opacity="0.10"/>` +
    `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2"/>`;
}

function renderWeeks(weeks) {
  const tb = $("rp-weeks");
  if (!weeks.length) {
    tb.innerHTML = `<tr><td colspan="5" class="rp-empty">No weekly snapshots stored yet — they accumulate as you view reports each week.</td></tr>`;
    return;
  }
  tb.innerHTML = weeks.map((w) => {
    const net = Number(w.total_r);
    return `<tr>
      <td>${w.week || "—"}</td>
      <td>${w.trades}</td>
      <td>${w.win_rate}%</td>
      <td class="${net >= 0 ? "win" : "loss"}">${fmtR(net)}</td>
      <td class="loss">${Number(w.max_drawdown_r).toFixed(2)}R</td>
    </tr>`;
  }).join("");
}

function renderTrades(trades) {
  const tb = $("rp-trades");
  if (!trades.length) {
    tb.innerHTML = `<tr><td colspan="8" class="rp-empty">No trades in this window.</td></tr>`;
    return;
  }
  tb.innerHTML = trades.map((t) => {
    const rv = t.r_gross != null ? t.r_gross : t.r;
    const win = rv > 0;
    return `<tr>
      <td>${dt(t.entry_time)}</td>
      <td class="side-${t.side}">${t.side === "long" ? "▲ LONG" : "▼ SHORT"}</td>
      <td>${fmtPx(t.entry)}</td>
      <td>${fmtPx(t.sl)}</td>
      <td>${fmtPx(t.tp)}</td>
      <td>${fmtPx(t.exit)}</td>
      <td class="${win ? "win" : "loss"}">${fmtR(rv)}</td>
      <td class="${win ? "win" : "loss"}">${win ? "WIN" : "LOSS"}</td>
    </tr>`;
  }).join("");
}

/* ---- events ---- */
$("rp-symbol").addEventListener("change", load);
$("rp-days").addEventListener("change", load);
document.querySelectorAll("#rp-tf button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("#rp-tf button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    TF = parseInt(b.dataset.tf, 10);
    load();
  }));

/* The Symbol/Timeframe/Window controls live in the header; move the interactive
   backtest section right under them so changing a control visibly updates the very
   next block (the fixed 4H Portfolio / Forward cards follow). */
function reorderForControls() {
  const nav = document.querySelector(".rp-nav");
  const h2 = $("rp-bt-title"), note = $("rp-bt-note"),
        status = $("rp-status"), body = $("rp-body");
  if (nav && h2 && note && status && body) nav.after(h2, note, status, body);
}

(async () => {
  reorderForControls();
  await loadSymbols();
  load();
  loadSignals();
  loadPortfolio();
  loadForward();
  setInterval(loadSignals, 60000);                       // scanner every minute
  setInterval(() => { loadPortfolio(); loadForward(); }, 300000);   // the rest every 5 min
})();
