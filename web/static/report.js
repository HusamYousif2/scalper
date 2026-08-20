/* report.js — the Live Performance page. Reads the live decision scorecard
   (/api/decisions) and the opportunity scanner (/api/signals). No backtest. */

const $ = (id) => document.getElementById(id);
const COIN = (s) => (s || "").replace("USDT", "");

function fmtPx(v) {
  if (v == null) return "—";
  const a = Math.abs(v);
  const dp = a >= 1000 ? 1 : a >= 1 ? 2 : 5;
  return v.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
}
function fmtR(v) { v = Number(v || 0); return (v >= 0 ? "+" : "") + v.toFixed(2) + "R"; }
function tfLabel(tf) { return tf >= 1440 ? "1D" : tf >= 60 ? `${tf / 60}H` : `${tf}m`; }
function clock() {
  const n = new Date();
  return `${String(n.getHours()).padStart(2, "0")}:${String(n.getMinutes()).padStart(2, "0")}`;
}
function ago(ts) {
  if (!ts) return "—";
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (s < 90) return `${s}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 172800) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
function since(ts) {
  if (!ts) return "just now";
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
const MEASURE_MIN = 50;   // don't headline a rate until the sample is big enough
                          // to be stable — with fewer calls, each new one moves
                          // the % too much and the number can't be trusted

/* ================= LIVE DECISION SCORECARD ================= */

async function loadDecisions() {
  try {
    const d = await fetch("/api/decisions").then((r) => {
      if (!r.ok) throw new Error(r.statusText); return r.json();
    });
    renderDecisions(d);
  } catch (e) { $("dec-badge").textContent = "warming up…"; }
}

function outMeta(s) {
  return {
    win: ["win", "TARGET ✓"], loss: ["loss", "STOP ✗"],
    expired_win: ["drift", "drifted +"], expired_loss: ["drift", "drifted −"],
  }[s] || ["drift", s];
}
function accClass(a) { return a == null ? "bad" : a >= 57 ? "good" : a >= 52.5 ? "mid" : "bad"; }

function renderDecisions(d) {
  const o = d.overall || {};
  const resolved = o.resolved || 0;
  const measuring = resolved < MEASURE_MIN;
  const good = o.win_rate >= 50;

  $("dec-updated").textContent = `live · updated ${clock()}`;
  $("hero-eyebrow").textContent = d.tracking_since
    ? `Live · since ${since(d.tracking_since)}`
    : "Live";

  // hero — headline rate only once we have enough settled calls
  const win = $("dec-win");
  if (measuring) {
    win.textContent = "measuring…";
    win.className = "hs-val mut";
    $("dec-winsub").textContent = resolved
      ? `${resolved} settled · need ${MEASURE_MIN}+ for stable %`
      : "first calls settling — check back in a few hours";
  } else {
    win.textContent = `${o.win_rate}%`;
    win.className = `hs-val ${good ? "up" : "down"}`;
    $("dec-winsub").textContent = `${o.wins} of ${resolved} calls`;
  }
  $("dec-hit").textContent = resolved ? `${o.target_hit_rate}%` : "—";
  const hi = (d.by_conviction || []).find((b) => b.min === 5);
  const hiEl = $("dec-hiwin");
  hiEl.textContent = hi && hi.resolved ? `${hi.win_rate}%` : "—";
  hiEl.className = `hs-val ${hi && hi.resolved ? "up" : "mut"}`;
  $("dec-opennow").textContent = o.open || 0;
  $("dec-resolved").textContent = `watching now · ${resolved} settled`;

  // recent-card badge
  const wb = $("dec-badge");
  if (!resolved) { wb.textContent = "warming up…"; wb.className = "win-badge"; }
  else { wb.textContent = `net ${fmtR(o.net_r)} · ${resolved} settled`;
         wb.className = `win-badge ${o.net_r >= 0 ? "good" : "bad"}`; }

  // live open calls
  const ob = $("open-badge");
  const opens = d.open_calls || [];
  ob.textContent = opens.length ? `${opens.length} in flight` : "none open yet";
  ob.className = `win-badge ${opens.length ? "good" : ""}`;
  $("open-rows").innerHTML = opens.length ? opens.map((r) =>
    `<tr>
       <td><b>${COIN(r.symbol)}</b></td><td>${tfLabel(r.tf)}</td>
       <td class="${r.bias === "long" ? "up" : "down"}">${r.bias === "long" ? "▲ LONG" : "▼ SHORT"}</td>
       <td>${r.passed}/${r.total_checks}</td>
       <td>${fmtPx(r.entry)}</td><td class="up">${fmtPx(r.target)}</td><td class="down">${fmtPx(r.stop)}</td>
       <td class="muted">${ago(r.opened_at)}</td>
     </tr>`).join("")
    : `<tr><td colspan="8"><div class="empty-note">No open calls yet — the moment a read
        fires on any coin/timeframe it appears here and the clock starts. First calls
        typically settle within the hour.</div></td></tr>`;

  // learned weights
  const wrap = $("dec-weights");
  const ws = (d.weights || []).filter((w) => w.accuracy != null);
  if (!ws.length) {
    wrap.innerHTML = `<div class="empty-note">Calibrating — the tool needs a batch of
      settled calls before it can measure each indicator and re-weight its vote.</div>`;
  } else {
    wrap.innerHTML = ws.map((w) => {
      const acc = w.accuracy, cls = accClass(acc);
      const pct = acc == null ? 0 : Math.max(4, Math.min(100, (acc - 45) / 20 * 100));
      return `<div class="wt-row">
        <div>
          <div class="wt-name">${w.indicator}</div>
          <div class="wt-bar"><span class="wt-fill ${cls}" style="width:${pct}%"></span></div>
          <div class="wt-meta">weight ${Number(w.weight).toFixed(2)} · ${w.n || 0} calls scored</div>
        </div>
        <div class="wt-acc ${cls}">${acc + "%"}</div>
      </div>`;
    }).join("");
  }

  // confidence
  $("dec-conv").innerHTML = (d.by_conviction || []).map((b) =>
    `<tr><td>${b.band}</td><td>${b.resolved}</td>
       <td>${b.win_rate}%</td><td>${b.target_hit_rate}%</td>
       <td class="${b.net_r >= 0 ? "up" : "down"}">${fmtR(b.net_r)}</td></tr>`).join("")
    || `<tr class="muted"><td colspan="5">warming up…</td></tr>`;

  // by timeframe
  $("dec-tf").innerHTML = (d.by_tf || []).filter((t) => t.resolved).map((t) =>
    `<tr><td>${tfLabel(t.tf)}</td><td>${t.resolved}</td>
       <td>${t.win_rate}%</td><td>${t.target_hit_rate}%</td>
       <td class="${t.net_r >= 0 ? "up" : "down"}">${fmtR(t.net_r)}</td></tr>`).join("")
    || `<tr class="muted"><td colspan="5">warming up…</td></tr>`;

  // by coin
  $("dec-coin").innerHTML = (d.by_symbol || []).filter((s) => s.resolved)
    .sort((a, b) => b.win_rate - a.win_rate).map((s) =>
    `<tr><td>${COIN(s.symbol)}</td><td>${s.resolved}</td>
       <td>${s.win_rate}%</td><td>${s.target_hit_rate}%</td>
       <td class="${s.net_r >= 0 ? "up" : "down"}">${fmtR(s.net_r)}</td></tr>`).join("")
    || `<tr class="muted"><td colspan="5">warming up…</td></tr>`;

  // recent resolved
  $("dec-recent").innerHTML = (d.recent || []).slice(0, 30).map((r) => {
    const [cls, lab] = outMeta(r.status);
    return `<tr>
      <td><b>${COIN(r.symbol)}</b></td><td>${tfLabel(r.tf)}</td>
      <td class="${r.bias === "long" ? "up" : "down"}">${r.bias === "long" ? "▲ LONG" : "▼ SHORT"}</td>
      <td>${r.passed}/${r.total_checks}</td>
      <td>${fmtPx(r.entry)}</td><td>${fmtPx(r.target)}</td>
      <td><span class="pill ${cls}">${lab}</span></td>
      <td class="${(r.outcome_r || 0) >= 0 ? "up" : "down"}">${fmtR(r.outcome_r || 0)}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="8"><div class="empty-note">Nothing has settled yet.
      As soon as an open call reaches its target or stop, it lands here — frozen.</div></td></tr>`;
}

/* ================= OPPORTUNITY SCANNER ================= */

async function loadSignals() {
  try {
    const j = await fetch("/api/signals?tf=240").then((r) => r.json());
    renderSignals(j.rows || []);
  } catch (e) { /* keep last render */ }
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
  if (!rows.length) { tb.innerHTML = `<tr class="muted"><td colspan="8">Scanning…</td></tr>`; return; }
  const fires = rows.filter((r) => r.signal_now).length;
  const strong = rows.filter((r) => r.score >= 72).length;
  const badge = $("sig-badge");
  badge.textContent = fires ? `${fires} firing now` : strong ? `${strong} strong` : "watching";
  badge.className = `win-badge ${fires ? "good" : ""}`;
  $("sig-updated").textContent = `4H · updated ${clock()}`;

  const t = rows[0], top = $("sig-top");
  if (top && t) {
    const col = ratingColor(t.rating);
    top.innerHTML = `<div class="sig-top ${t.bias}">
      <div class="sig-top-l">
        <div class="sig-top-coin">${COIN(t.symbol)}<span> / USDT</span></div>
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
      <td>${i + 1}</td><td><b>${COIN(r.symbol)}</b></td>
      <td class="sig-score">${scoreBar(r.score, r.rating)}</td>
      <td>${biasTag(r.bias)}</td>
      <td style="color:${ratingColor(r.rating)}">${live ? "● FIRE" : r.rating}</td>
      <td>${fmtPx(r.entry)}</td><td class="loss">${fmtPx(r.stop)}</td><td class="win">${fmtPx(r.target)}</td>
    </tr>`;
  }).join("");
}

/* ================= init ================= */

(async () => {
  loadDecisions();
  loadSignals();
  setInterval(loadSignals, 60000);        // scanner every minute
  setInterval(loadDecisions, 300000);     // scorecard every 5 min — stable, not jumpy
})();
