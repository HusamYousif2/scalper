/* glossary.js — every term the interface uses, defined in plain English.
   Reachable two ways: the drawer in the header, and the small "?" next to any
   term in the page, which opens the drawer scrolled to that entry. */

const GLOSSARY = [
  {
    id: "bps",
    term: "Basis point (bps)",
    short: "One hundredth of one percent.",
    body: `100 bps = 1%. 10 bps = 0.1%. 1 bps = 0.01%.
      <br><br>Traders use basis points because intraday moves are small and
      writing "0.075%" over and over is awkward. On a $64,000 Bitcoin price,
      10 bps is about $64.`,
    example: "A 15 bps move on BTC at $64,000 ≈ $96 per coin.",
  },
  {
    id: "percentile",
    term: "Percentile",
    short: "Where today's reading ranks against recent history.",
    body: `If volatility is at the <b>70th percentile</b>, it means that over the
      last 30 days, 70% of moments were calmer than right now and 30% were
      wilder.
      <br><br>50th percentile = perfectly average. 95th = one of the most active
      moments of the month. 5th = unusually dead.`,
    example: "Volatility rank 82nd → busier than 82% of the last month.",
  },
  {
    id: "roundtrip",
    term: "Round trip",
    short: "The full cost of opening and then closing a position.",
    body: `You pay a fee to enter and another to exit. If you use market orders
      you also pay the spread — the gap between the best bid and the best ask.
      <br><br>Add them together and that is your round trip. It is the hurdle
      every trade has to clear before a single dollar is yours.`,
    example: "Binance futures taker: 0.045% in + 0.045% out + spread ≈ 10.5 bps.",
  },
  {
    id: "maker-taker",
    term: "Maker vs taker",
    short: "Whether your order waits in the book or crosses it.",
    body: `A <b>maker</b> order rests in the order book and waits for someone to
      trade against it. It is cheaper, but it may never fill.
      <br><br>A <b>taker</b> order crosses the spread and fills immediately at
      whatever is available. It costs more, and you also give up the spread.
      <br><br>For scalping this choice can be the difference between a profitable
      and an unprofitable system, because the cost difference is larger than the
      typical edge.`,
    example: "Futures maker ≈ 4 bps round trip. Futures taker ≈ 10.5 bps.",
  },
  {
    id: "aggressor",
    term: "Aggressor balance",
    short: "Who is crossing the spread to get filled — buyers or sellers.",
    body: `Every trade has a passive side and an aggressive side. The aggressor is
      whoever was impatient enough to cross the spread.
      <br><br>+1.0 means every trade in the window was a buyer lifting the offer.
      −1.0 means every trade was a seller hitting the bid. 0 means balanced.
      <br><br>This cannot be read from a candlestick chart — it needs the raw
      trade tape, which is what this desk processes.`,
    example: "+0.30 → clearly more aggressive buying than selling.",
  },
  {
    id: "largeprint",
    term: "Large-print flow",
    short: "The same balance, but only counting orders above $100,000.",
    body: `Small orders are mostly noise: retail, bots, order slicing. Orders
      above $100,000 usually reflect a decision by someone with size.
      <br><br>When large prints lean one way while the overall tape leans the
      other, the two groups disagree.`,
    example: "Large-print flow +0.35 while total flow is 0.00 → size is buying quietly.",
  },
  {
    id: "openinterest",
    term: "Open interest",
    short: "The total value of futures positions currently open.",
    body: `Rising open interest means new positions are being opened. Falling open
      interest means positions are being closed.
      <br><br>Combined with price it is informative: price up with open interest
      up means new longs are entering. Price up with open interest down means
      shorts are covering — a very different situation.`,
    example: "OI +0.4% with price rising → new money entering long.",
  },
  {
    id: "longshort",
    term: "Long/short ratio",
    short: "How much money is positioned long versus short.",
    body: `A ratio of 2.0 means twice as much is long as short. 0.5 means twice as
      much is short.
      <br><br>We show it separately for retail accounts and for the exchange's
      largest accounts by position size. When the two diverge sharply, one group
      is on the wrong side of the trade.`,
    example: "Retail 2.4 : 1 while top accounts sit at 1.0 : 1 → crowded retail long.",
  },
  {
    id: "amplification",
    term: "Amplification",
    short: "How much bigger the move that follows a reading tends to be.",
    body: `1.00× means moves following readings like this one are normal for this
      market. 1.40× means they are 40% larger than normal. 0.80× means 20%
      smaller.
      <br><br>It is measured from history: we bucket every past reading of that
      indicator, then measure the move that actually followed each bucket.
      <br><br>It says nothing about direction — only about size.`,
    example: "ATR at the 90th percentile → 1.35× → expect a larger than usual move.",
  },
  {
    id: "efficiency",
    term: "Efficiency ratio",
    short: "How much ground price covered versus how far it actually got.",
    body: `Take the net distance travelled and divide it by the total path length.
      <br><br>Near <b>1.0</b>: price moved in a straight line — a trend.
      <br>Near <b>0.0</b>: price covered a lot of ground and ended up where it
      started — churn.
      <br><br>Breakout methods work in the first case and bleed in the second.
      Fade methods do the reverse.`,
    example: "0.05 over four hours → heavy churn, poor conditions for breakouts.",
  },
  {
    id: "quantile",
    term: "Quantile",
    short: "A level with a stated frequency attached.",
    body: `The <b>90th quantile</b> of the coming move is the distance that 9 out
      of 10 moves stay below. The <b>median</b> (50th) is the midpoint: half the
      moves are smaller, half are larger.
      <br><br>This is more useful than a single average, because it tells you the
      shape of what can happen — which is exactly what a stop-loss has to survive.`,
    example: "90th quantile = 21 bps → a stop at 21 bps is hit by 1 move in 10.",
  },
  {
    id: "calibration",
    term: "Calibration",
    short: "Whether a stated probability is actually true.",
    body: `A forecast that says "90% of moves stay under 20 bps" is calibrated if,
      in reality, about 90% of moves stay under 20 bps.
      <br><br>Most tools never check this. It is the difference between a number
      you can size a position with and a number that is decoration.
      <br><br>This desk's quantiles are calibrated to within 0.8 percentage
      points, measured across 17,894 forecasts on data the model never saw.`,
    example: "Promised 90%, delivered 90.2% → calibrated.",
  },
  {
    id: "har",
    term: "HAR-RV benchmark",
    short: "The published academic standard for volatility forecasting.",
    body: `HAR-RV (Corsi, 2009) forecasts volatility from three lagged averages —
      recent, medium and long. It is simple, and it is famously difficult to beat;
      a great many published models fail to.
      <br><br>We show our score next to it on every screen so the improvement is
      verifiable rather than asserted.`,
    example: "Benchmark 0.232, this model 0.355 → 53% better.",
  },
  {
    id: "r2",
    term: "R² (out-of-sample)",
    short: "How much of the variation the model explains, on unseen data.",
    body: `R² runs from 0 to 1. Zero means the model does no better than always
      guessing the average. One means perfect prediction.
      <br><br>"Out-of-sample" is the critical part: the score is measured on data
      the model was never trained on. In-sample scores can be made arbitrarily
      high and mean nothing.
      <br><br>For volatility forecasting, anything above 0.30 out-of-sample is a
      strong result.`,
    example: "0.355 out-of-sample on BTC 15-minute forecasts.",
  },
  {
    id: "poc",
    term: "Volume point of control",
    short: "The price level where the most volume changed hands.",
    body: `Over any window, some price levels see far more trading than others.
      The point of control is the busiest one.
      <br><br>Price tends to be drawn back toward it, and to react when it
      arrives, because that is where the largest number of positions were opened.`,
    example: "Price 40 bps above the daily point of control → stretched.",
  },
  {
    id: "atr",
    term: "Average True Range (ATR)",
    short: "The average distance price covers per candle.",
    body: `It measures how wide recent candles have been, including gaps. It is
      the most direct measure of how much a market is moving.
      <br><br>In our testing ATR was the single most informative technical
      reading for forecasting the size of the next move — more than any momentum
      indicator.`,
    example: "ATR at the 85th percentile → an unusually wide-ranging market.",
  },
  {
    id: "microstructure",
    term: "Market microstructure",
    short: "The mechanics underneath the price: orders, fills, and the book.",
    body: `Candlestick charts summarise what happened. Microstructure is the raw
      material they were built from — every individual trade, its size, its
      direction, and the resting orders around it.
      <br><br>This desk processes roughly 460,000 individual trades per day per
      market, plus order book depth, open interest and positioning.`,
    example: "One day of BTC futures ≈ 460,000 trades ≈ 6 MB compressed.",
  },
  {
    id: "walkforward",
    term: "Walk-forward testing",
    short: "Training on the past, then testing only on what came after.",
    body: `The model is trained on a window of history, then scored on the period
      immediately following it — which it has never seen. Then the window rolls
      forward and the process repeats.
      <br><br>This is how a model gets tested the way it will actually be used.
      A backtest that trains and tests on the same period can be made to look
      spectacular and will fail in live markets.`,
    example: "Retrained daily, scored on the next day, repeated across two years.",
  },
];

function renderGlossary(filter = "") {
  const q = filter.trim().toLowerCase();
  const items = GLOSSARY.filter(
    (g) =>
      !q ||
      g.term.toLowerCase().includes(q) ||
      g.short.toLowerCase().includes(q) ||
      g.body.toLowerCase().includes(q)
  );
  const body = document.getElementById("glossary-body");
  body.innerHTML = items
    .map(
      (g) => `<article class="g-entry" id="g-${g.id}">
        <h3>${g.term}</h3>
        <p class="g-short">${g.short}</p>
        <div class="g-body">${g.body}</div>
        ${g.example ? `<div class="g-example">${g.example}</div>` : ""}
      </article>`
    )
    .join("");
  if (!items.length) body.innerHTML = `<p class="g-empty">No matching term.</p>`;
}

function openGlossary(termId) {
  document.getElementById("drawer").hidden = false;
  document.getElementById("drawer-backdrop").hidden = false;
  document.body.style.overflow = "hidden";
  renderGlossary("");
  if (termId) {
    const el = document.getElementById(`g-${termId}`);
    if (el) {
      el.scrollIntoView({ block: "start" });
      el.classList.add("flash");
      setTimeout(() => el.classList.remove("flash"), 1600);
    }
  }
}

function closeGlossary() {
  document.getElementById("drawer").hidden = true;
  document.getElementById("drawer-backdrop").hidden = true;
  document.body.style.overflow = "";
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("open-glossary").addEventListener("click", () => openGlossary());
  document.getElementById("close-glossary").addEventListener("click", closeGlossary);
  document.getElementById("drawer-backdrop").addEventListener("click", closeGlossary);
  document.getElementById("glossary-search").addEventListener("input", (e) =>
    renderGlossary(e.target.value)
  );
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeGlossary();
  });
});

/* the small "?" markers scattered through the page */
document.addEventListener("click", (e) => {
  const q = e.target.closest(".q");
  if (q) openGlossary(q.dataset.term);
});

/* hover preview so a reader does not have to leave the page */
document.addEventListener("mouseover", (e) => {
  const q = e.target.closest(".q");
  if (!q) return;
  const g = GLOSSARY.find((x) => x.id === q.dataset.term);
  if (!g) return;
  const tip = document.getElementById("tooltip");
  tip.innerHTML = `<b>${g.term}</b><br>${g.short}
    <span class="tip-more">click for the full definition</span>`;
  tip.hidden = false;
  const r = q.getBoundingClientRect();
  tip.style.left = `${Math.min(r.left, window.innerWidth - 300)}px`;
  tip.style.top = `${r.bottom + window.scrollY + 8}px`;
});
document.addEventListener("mouseout", (e) => {
  if (e.target.closest(".q")) document.getElementById("tooltip").hidden = true;
});
