/* indicator_info.js — the reference card that opens when a trader clicks an
   indicator line or its name.

   Each entry carries: what the indicator is, the formula in readable notation,
   how to read it, and — for the proprietary studies — why it needs data a normal
   chart does not have. This is the "click the coloured line and see everything
   about it" behaviour from a full charting terminal. */

const INDICATOR_INFO = {
  ema9:  { name: "Exponential Moving Average (9)", group: "Trend",
    formula: "EMA_t = price_t · k + EMA_(t−1) · (1 − k),  k = 2 / (9 + 1)",
    what: "A fast-reacting average of the closing price over the last 9 candles.",
    read: "Price above the line is short-term bullish, below is bearish. The 9 reacts quickly, so it hugs price closely and is used for timing rather than trend." },
  ema21: { name: "Exponential Moving Average (21)", group: "Trend",
    formula: "EMA_t = price_t · k + EMA_(t−1) · (1 − k),  k = 2 / (21 + 1)",
    what: "A medium-speed average of closing price over 21 candles.",
    read: "The 21 is a common intraday trend line. When the 21 is above the 50, the shorter-term trend is up; the crossover of the two is a classic momentum signal." },
  ema50: { name: "Exponential Moving Average (50)", group: "Trend",
    formula: "EMA_t = price_t · k + EMA_(t−1) · (1 − k),  k = 2 / (50 + 1)",
    what: "A slower average that marks the intermediate trend.",
    read: "Price holding above the 50 keeps the intermediate trend intact. It often acts as dynamic support in an uptrend and resistance in a downtrend." },
  ema200: { name: "Exponential Moving Average (200)", group: "Trend",
    formula: "EMA_t = price_t · k + EMA_(t−1) · (1 − k),  k = 2 / (200 + 1)",
    what: "The long-term trend line watched by most of the market.",
    read: "Above the 200 is broadly bullish, below is broadly bearish. Because so many traders watch it, it often becomes self-fulfilling support or resistance." },
  sma20: { name: "Simple Moving Average (20)", group: "Trend",
    formula: "SMA = (P₁ + P₂ + … + P₂₀) / 20",
    what: "The plain average of the last 20 closes, every candle weighted equally.",
    read: "Smoother and slower than an EMA of the same length. It is the centre line of the Bollinger Bands." },
  vwap: { name: "VWAP — Volume-Weighted Average Price", group: "Trend",
    formula: "VWAP = Σ(price · volume) / Σ(volume),  reset each UTC day",
    what: "The average price paid today, weighted by how much traded at each level.",
    read: "The single most-watched intraday level for institutions. Price above VWAP means buyers today are in profit on average; below means sellers are. It often acts as a magnet and a fair-value line." },
  vwap_up: { name: "VWAP upper band (+1σ)", group: "Trend",
    formula: "VWAP + 1 standard deviation of price around VWAP",
    what: "One standard deviation above the volume-weighted average.",
    read: "Price reaching the upper band is stretched above fair value for the day — a common area for mean-reversion sellers." },
  vwap_dn: { name: "VWAP lower band (−1σ)", group: "Trend",
    formula: "VWAP − 1 standard deviation of price around VWAP",
    what: "One standard deviation below the volume-weighted average.",
    read: "Price at the lower band is stretched below fair value — a common area for dip buyers." },
  bb_up: { name: "Bollinger Band — upper", group: "Volatility",
    formula: "SMA(20) + 2 · standard deviation(20)",
    what: "Two standard deviations above the 20-period average.",
    read: "Price tagging the upper band shows strong upward momentum, but also a stretched market. Bands widening means volatility is rising; narrowing (a 'squeeze') often precedes a large move." },
  bb_mid: { name: "Bollinger Band — basis", group: "Volatility",
    formula: "SMA(20)",
    what: "The middle Bollinger line, a simple 20-period average.",
    read: "Acts as a mean that price reverts to inside a range." },
  bb_dn: { name: "Bollinger Band — lower", group: "Volatility",
    formula: "SMA(20) − 2 · standard deviation(20)",
    what: "Two standard deviations below the 20-period average.",
    read: "Price at the lower band shows strong downward momentum in a stretched market." },
  kc_up: { name: "Keltner Channel — upper", group: "Volatility",
    formula: "EMA(20) + 1.5 · ATR(20)",
    what: "An ATR-based envelope above the average price.",
    read: "Unlike Bollinger Bands, Keltner uses the average true range, so it reacts to real range rather than closing-price scatter. When Bollinger Bands sit inside the Keltner channel, it signals a volatility squeeze." },
  kc_dn: { name: "Keltner Channel — lower", group: "Volatility",
    formula: "EMA(20) − 1.5 · ATR(20)",
    what: "An ATR-based envelope below the average price.",
    read: "Used with the upper band to gauge whether the market is compressed or expanding." },
  dc_up: { name: "Donchian Channel — high", group: "Volatility",
    formula: "highest high over the last 20 candles",
    what: "The top of the recent trading range.",
    read: "A break above the Donchian high is the classic breakout signal used by trend-following systems." },
  dc_dn: { name: "Donchian Channel — low", group: "Volatility",
    formula: "lowest low over the last 20 candles",
    what: "The bottom of the recent trading range.",
    read: "A break below the Donchian low is a classic breakdown signal." },
  supertrend: { name: "SuperTrend", group: "Trend",
    formula: "flips at  (high+low)/2  ±  3 · ATR(10),  trailing",
    what: "A trailing stop-and-reverse line based on the average true range.",
    read: "Green line below price = uptrend; red line above price = downtrend. The line itself is a ready-made trailing stop level." },
  tenkan: { name: "Ichimoku — Tenkan-sen (Conversion)", group: "Trend",
    formula: "(9-period high + 9-period low) / 2",
    what: "The fast line of the Ichimoku system.",
    read: "A short-term equilibrium price. A Tenkan cross above the Kijun is a bullish momentum signal." },
  kijun: { name: "Ichimoku — Kijun-sen (Base)", group: "Trend",
    formula: "(26-period high + 26-period low) / 2",
    what: "The medium-term equilibrium line of the Ichimoku system.",
    read: "Acts as support/resistance and a trailing reference; price tends to revert to it." },
  senkou_a: { name: "Ichimoku — Leading Span A", group: "Trend",
    formula: "(Tenkan + Kijun) / 2, plotted 26 candles ahead",
    what: "One edge of the Ichimoku cloud.",
    read: "The gap between Span A and Span B forms the cloud. Price above the cloud is bullish, below is bearish, inside is undecided." },
  senkou_b: { name: "Ichimoku — Leading Span B", group: "Trend",
    formula: "(52-period high + 52-period low) / 2, plotted 26 candles ahead",
    what: "The other edge of the Ichimoku cloud.",
    read: "A thick cloud means strong support/resistance; a thin cloud is easily broken." },
  psar: { name: "Parabolic SAR", group: "Trend",
    formula: "SAR_t = SAR_(t−1) + AF · (EP − SAR_(t−1))",
    what: "A stop-and-reverse dot that accelerates as a trend extends.",
    read: "Dots below price signal an uptrend, above price a downtrend. When the dots flip sides, the trend may be reversing. Often used as a trailing stop." },

  volume: { name: "Volume", group: "Volume",
    formula: "total base-asset volume traded per candle",
    what: "How much was traded in each candle, coloured by whether the candle closed up or down.",
    read: "Rising volume confirms a move; a breakout on low volume is suspect. Volume spikes often mark exhaustion or the start of a new move." },
  rsi: { name: "RSI — Relative Strength Index (14)", group: "Momentum",
    formula: "RSI = 100 − 100 / (1 + avg gain / avg loss),  over 14 candles",
    what: "A 0–100 momentum oscillator measuring the speed of recent gains against losses.",
    read: "Above 70 is traditionally overbought, below 30 oversold. In a strong trend it can stay stretched for a long time, so it is best used for divergences rather than as an automatic reversal signal." },
  stoch_k: { name: "Stochastic %K (14)", group: "Momentum",
    formula: "%K = 100 · (close − low₁₄) / (high₁₄ − low₁₄)",
    what: "Where the close sits within the recent high-low range.",
    read: "Above 80 is overbought, below 20 oversold. The %K crossing the %D line is the trade trigger. Works best in ranging markets." },
  stoch_d: { name: "Stochastic %D", group: "Momentum",
    formula: "%D = 3-period average of %K",
    what: "The signal line of the stochastic oscillator.",
    read: "Crossovers of %K and %D generate the signals." },
  stochrsi_k: { name: "Stochastic RSI", group: "Momentum",
    formula: "the Stochastic formula applied to RSI instead of price",
    what: "A more sensitive oscillator — the stochastic of the RSI.",
    read: "Reaches its extremes far more often than plain RSI, so it gives earlier but noisier signals." },
  macd: { name: "MACD line", group: "Momentum",
    formula: "EMA(12) − EMA(26)",
    what: "The gap between a fast and a slow moving average — a momentum measure.",
    read: "Above zero is bullish momentum, below zero bearish. The MACD crossing its signal line is the classic entry trigger." },
  macd_signal: { name: "MACD signal line", group: "Momentum",
    formula: "EMA(9) of the MACD line",
    what: "A smoothed version of the MACD used as a trigger line.",
    read: "MACD crossing above the signal is bullish; crossing below is bearish." },
  macd_hist: { name: "MACD histogram", group: "Momentum",
    formula: "MACD line − signal line",
    what: "The distance between the MACD and its signal, drawn as bars.",
    read: "Bars growing = momentum accelerating; bars shrinking = momentum fading, often before the lines actually cross." },
  adx: { name: "ADX — Average Directional Index (14)", group: "Trend strength",
    formula: "smoothed average of |DI+ − DI−| / (DI+ + DI−)",
    what: "Measures how strong a trend is, regardless of direction.",
    read: "Above 25 means a real trend is present; below 20 means the market is ranging. It says nothing about direction — pair it with DI+ and DI−." },
  di_plus: { name: "DI+ (positive directional)", group: "Trend strength",
    formula: "smoothed upward movement / average true range",
    what: "The strength of upward movement.",
    read: "When DI+ is above DI−, buyers are in control." },
  di_minus: { name: "DI− (negative directional)", group: "Trend strength",
    formula: "smoothed downward movement / average true range",
    what: "The strength of downward movement.",
    read: "When DI− is above DI+, sellers are in control." },
  cci: { name: "CCI — Commodity Channel Index (20)", group: "Momentum",
    formula: "(typical price − SMA) / (0.015 · mean deviation)",
    what: "How far price has strayed from its statistical average.",
    read: "Above +100 is strong (possibly overbought), below −100 is weak (possibly oversold). Good for spotting the start of new moves." },
  williams_r: { name: "Williams %R (14)", group: "Momentum",
    formula: "−100 · (high₁₄ − close) / (high₁₄ − low₁₄)",
    what: "An inverted stochastic, scaled −100 to 0.",
    read: "Above −20 is overbought, below −80 oversold. Reacts fast, so it is used for timing entries." },
  mfi: { name: "MFI — Money Flow Index (14)", group: "Volume",
    formula: "RSI, but weighted by volume instead of price alone",
    what: "A volume-weighted momentum oscillator — 'RSI with volume'.",
    read: "Above 80 overbought, below 20 oversold. Because it uses volume, divergences on the MFI carry more weight than on the plain RSI." },
  atr_pct: { name: "ATR % — Average True Range", group: "Volatility",
    formula: "ATR(14) / price · 100",
    what: "The average candle range as a percentage of price — a pure volatility gauge.",
    read: "Rising ATR% means bigger swings and wider stops needed; falling ATR% means a calming market. This is the raw material behind the move-size forecast on this page." },
  obv: { name: "OBV — On-Balance Volume", group: "Volume",
    formula: "running total: + volume on up candles, − volume on down candles",
    what: "A cumulative volume line that rises on up candles and falls on down candles.",
    read: "If OBV makes a new high before price does, buying pressure is leading — a bullish tell. Divergence between OBV and price warns of a weakening move." },
  roc: { name: "ROC — Rate of Change (12)", group: "Momentum",
    formula: "(price − price₁₂) / price₁₂ · 100",
    what: "The percentage change over the last 12 candles.",
    read: "Simple momentum: positive and rising is accelerating up, negative and falling is accelerating down. Zero-line crosses mark momentum shifts." },

  cvd: { name: "Cumulative Volume Delta", group: "ASTRA — order flow", proprietary: true,
    formula: "running total of (aggressive buy volume − aggressive sell volume)",
    what: "Tracks whether market orders are net buying or net selling, cumulatively.",
    read: "Rising CVD while price rises confirms real buying. Rising price with falling CVD is a warning — the move is not backed by aggressive buyers and may fail. This needs the raw trade tape, which candlesticks do not contain." },
  delta: { name: "Volume Delta", group: "ASTRA — order flow", proprietary: true,
    formula: "aggressive buy volume − aggressive sell volume, per candle",
    what: "The net aggression in each individual candle.",
    read: "Large positive bars are bursts of market buying, large negative bars bursts of selling. A big delta that fails to move price signals absorption by a large passive order." },
  aggressor: { name: "Aggressor Imbalance", group: "ASTRA — order flow", proprietary: true,
    formula: "(buy − sell) / (buy + sell), over a rolling window",
    what: "Who is crossing the spread to get filled, from −1 (all sellers) to +1 (all buyers).",
    read: "Sustained positive readings mean impatient buyers are driving price. This is a live read of pressure that no price-only indicator can see." },
  whale_flow: { name: "Whale Flow", group: "ASTRA — order flow", proprietary: true,
    formula: "buy/sell imbalance counting only trades above $100,000",
    what: "The direction of large orders only, ignoring retail-sized noise.",
    read: "When whales lean one way while the broad tape leans the other, size is usually right. Divergence between whale flow and price is one of the strongest tells this terminal produces." },
  whale_share: { name: "Whale Share of Volume", group: "ASTRA — order flow", proprietary: true,
    formula: "large-trade volume / total volume",
    what: "How much of the current activity is coming from big players.",
    read: "A high share means the move is institution-driven; a low share means it is retail churn." },
  absorption: { name: "Absorption", group: "ASTRA — order flow", proprietary: true,
    formula: "price movement per unit of one-sided aggression (standardised)",
    what: "Detects when heavy one-way pressure fails to move price — a large passive order is soaking it up.",
    read: "High absorption often precedes a sharp move: once the hidden order is filled, the wall is gone and price snaps. A classic sign of a large player positioning quietly." },
  intensity: { name: "Trade Intensity", group: "ASTRA — order flow", proprietary: true,
    formula: "recent trade count / daily-average trade count",
    what: "How busy the tape is right now versus a normal moment.",
    read: "Above 1 means unusual activity — moves during high-intensity periods tend to be larger and more decisive." },
  oi_chg: { name: "Open Interest change", group: "ASTRA — derivatives", proprietary: true,
    formula: "percentage change in total open futures positions",
    what: "Whether traders are opening new positions or closing existing ones.",
    read: "Price up + OI up = new longs, a move with fuel. Price up + OI down = shorts covering, a move that may fade. This comes from the futures endpoint, not the chart." },
  crowd_gap: { name: "Retail vs Smart Money", group: "ASTRA — positioning", proprietary: true,
    formula: "log(retail long/short) − log(top-trader long/short)",
    what: "The gap between how retail is positioned and how the largest accounts are positioned.",
    read: "When the crowd is heavily long while big accounts are not, it warns of a crowded trade vulnerable to a squeeze. A contrarian's gauge." },

  /* ---------- your strategy (the pro 4H setup) ---------- */
  pro_t3: { name: "T3 (8, 0.7) — trend permission", group: "Your strategy",
    formula: "6-times-smoothed EMA (Tillson T3)",
    what: "A smooth trend line. It is the strategy's gatekeeper: it only allows longs when price is above it, and shorts when price is below it.",
    read: "Price above a rising T3 = uptrend. Price below a falling T3 = downtrend. A flat T3 means no trend — stand aside.",
    buy: "Only buy when price is ABOVE the T3 line and the line is sloping UP. If price closes back below T3, exit the long.",
    sell: "Only sell/short when price is BELOW the T3 line and it is sloping DOWN. If price closes back above T3, exit the short." },
  pro_rf: { name: "Range Filter (50, 3.0)", group: "Your strategy",
    formula: "smoothed average range band that only moves when price breaks it",
    what: "A stair-step line that stays flat during noise and steps up or down in a real trend — it filters out chop.",
    read: "A rising filter with price above it = healthy uptrend. Falling filter with price below = downtrend.",
    buy: "Buy when the filter turns UP and price sits above it (a fresh up-step). This is the strategy's 'entry event A'.",
    sell: "Sell when the filter turns DOWN and price sits below it (a fresh down-step)." },
  pro_sq: { name: "Squeeze Momentum", group: "Your strategy",
    formula: "linear-regression momentum + Bollinger/Keltner squeeze state (TTM)",
    what: "Measures momentum after a quiet 'squeeze'. Green bars above zero = up-momentum, red bars below zero = down-momentum. When a squeeze releases, a big move usually follows.",
    read: "Bars growing above zero = buyers taking over. Bars growing below zero = sellers taking over.",
    buy: "Buy when the bars cross ABOVE zero (turn green), ideally right as a squeeze releases — 'entry event B'.",
    sell: "Sell when the bars cross BELOW zero (turn red)." },
  pro_cov: { name: "Change of Volatility — the edge", group: "Your strategy",
    formula: "short-term volatility ÷ its own longer baseline − 1",
    what: "Tells you whether the market is WAKING UP. Above zero and rising means volatility is expanding — the fuel a breakout needs.",
    read: "Rising and positive = a real move is starting. Falling / below zero = a dead, choppy market.",
    buy: "Not a buy signal by itself. Only take your buy (from T3 / Range Filter / Squeeze) WHEN this is positive and rising.",
    sell: "Same for sells — only short when volatility is expanding. When this is falling, skip the trade entirely." },
  pro_cvol: { name: "Chaikin Volatility — the edge", group: "Your strategy",
    formula: "rate of change of the smoothed high-low range",
    what: "A second volatility-expansion gauge, from the candle range. Positive = the range is widening (a move is underway).",
    read: "Above zero = expanding volatility, good for breakouts. Below zero = contracting, avoid.",
    buy: "Confirmation only: take longs when this is positive (volatility expanding up).",
    sell: "Take shorts when this is positive alongside a down signal. Ignore signals when it is negative." },
  pro_adx: { name: "ADX (10) — trend strength", group: "Your strategy",
    formula: "smoothed |DI+ − DI−| ÷ (DI+ + DI−)",
    what: "How STRONG the trend is (not its direction). The strategy needs ADX above ~20 before it trusts any signal.",
    read: "Above 20 and rising = a trend with power. Below 20 = range / chop, where breakouts fail.",
    buy: "Not directional. Only take a buy when ADX is above ~20. Don't buy in a flat, low-ADX market.",
    sell: "Same — only short when ADX is above ~20." },
  pro_dip: { name: "DI + (10) — buyers' strength", group: "Your strategy",
    formula: "smoothed positive directional movement",
    what: "The green directional line. When DI+ is above DI−, buyers are in control.",
    read: "DI+ above DI− = upward pressure dominates.",
    buy: "Buy bias when DI+ crosses ABOVE DI− (with ADX > 20).",
    sell: "Exit longs / avoid buys when DI+ drops back below DI−." },
  pro_dim: { name: "DI − (10) — sellers' strength", group: "Your strategy",
    formula: "smoothed negative directional movement",
    what: "The red directional line. When DI− is above DI+, sellers are in control.",
    read: "DI− above DI+ = downward pressure dominates.",
    buy: "Avoid buys while DI− leads.",
    sell: "Short bias when DI− crosses ABOVE DI+ (with ADX > 20)." },
  pro_atr_pct: { name: "ATR % (14) — stop & target sizer", group: "Your strategy",
    formula: "Average True Range ÷ price × 100",
    what: "How much the coin typically moves, as a percent. The strategy uses it to place the stop and target — it is a risk tool, not a buy/sell signal.",
    read: "Bigger ATR% = bigger candles = wider stop needed. Tiny ATR% = dead market, skip.",
    buy: "After a buy signal, put your STOP about 2×ATR below entry and your first TARGET about 4×ATR above (≈ 2R).",
    sell: "After a sell signal, STOP about 2×ATR above entry, TARGET about 4×ATR below. Avoid trading when ATR% is very small." },
};

/* short buy/sell hints for the classic indicators (shown in the same card) */
const TRADE_HINTS = {
  ema21: { buy: "Buy when price is above the 21 and the 21 is above the 50 (short-term uptrend).", sell: "Sell when price falls below the 21 and it rolls under the 50." },
  ema50: { buy: "Buy pullbacks that hold above a rising 50 — it acts as support in an uptrend.", sell: "Sell rallies that fail under a falling 50 — it acts as resistance in a downtrend." },
  ema200: { buy: "Prefer longs while price is above the 200 (broad uptrend).", sell: "Prefer shorts while price is below the 200 (broad downtrend)." },
  vwap: { buy: "Buy when price is above VWAP (today's buyers are winning), or on a dip back to it that holds.", sell: "Sell when price is below VWAP, or on a bounce to it that fails." },
  bb_up: { buy: "In a strong trend, a close above the upper band is a breakout — buy continuation.", sell: "In a range, price tagging the upper band is stretched — a mean-reversion sell." },
  bb_dn: { buy: "In a range, price tagging the lower band is stretched — a mean-reversion buy.", sell: "In a downtrend, a close below the lower band is a breakdown — sell continuation." },
  supertrend: { buy: "Buy when SuperTrend flips GREEN and sits below price.", sell: "Sell when it flips RED and sits above price. Many traders trail their stop on this line." },
  rsi: { buy: "Buy as RSI turns up from below 30 (oversold). In an uptrend, use the 40 level as the dip zone.", sell: "Sell as RSI turns down from above 70 (overbought). In a downtrend, use 60 as the bounce zone." },
  macd: { buy: "Buy when MACD crosses above its signal line, especially below zero.", sell: "Sell when MACD crosses below its signal line, especially above zero." },
  macd_hist: { buy: "Buy when the histogram turns from red to green (momentum flipping up).", sell: "Sell when it turns from green to red (momentum flipping down)." },
  adx: { buy: "Only take buys when ADX is above ~20 (real trend). Low ADX = chop, stay out.", sell: "Only take shorts when ADX is above ~20." },
  di_plus: { buy: "Buy bias when DI+ crosses above DI−.", sell: "Reduce longs when DI+ falls back below DI−." },
  di_minus: { buy: "Avoid buys while DI− leads.", sell: "Short bias when DI− crosses above DI+." },
  cvd: { buy: "Buy when CVD is rising with price — real buying pressure backs the move.", sell: "Sell when CVD is falling with price. If price rises but CVD drops, the rally is weak (divergence)." },
  volume: { buy: "Trust an upside breakout that comes on a big volume bar.", sell: "Trust a breakdown on heavy volume; ignore moves on thin volume." },
  aggressor: { buy: "Buy when aggressive market buyers dominate (positive imbalance).", sell: "Sell when aggressive market sellers dominate (negative imbalance)." },
  whale_flow: { buy: "Buy alongside net whale buying — large players accumulating.", sell: "Sell alongside net whale selling — large players distributing." },
  oi_chg: { buy: "Price up + open interest up = new longs with fuel — buy continuation.", sell: "Price up + open interest down = short covering that may fade — avoid chasing." },
};

function showIndicatorInfo(key) {
  const info = INDICATOR_INFO[key];
  if (!info) return;
  const hint = TRADE_HINTS[key] || {};
  const buy = info.buy || hint.buy;
  const sell = info.sell || hint.sell;
  let modal = document.getElementById("ind-modal");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "ind-modal";
    modal.className = "ind-modal-backdrop";
    modal.innerHTML = `<div class="ind-modal" role="dialog"></div>`;
    document.body.appendChild(modal);
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.hidden = true;
    });
  }
  const val = document.getElementById(`pv-${key}`) || document.getElementById(`lgv-${key}`);
  const current = val && val.textContent ? val.textContent : null;
  const box = modal.querySelector(".ind-modal");
  box.innerHTML = `
    <div class="im-head">
      <div>
        <div class="im-group">${info.group}${info.proprietary
          ? ' <span class="badge-ai">AI</span>' : ""}</div>
        <h3>${info.name}</h3>
      </div>
      <button class="im-close">✕</button>
    </div>
    ${current ? `<div class="im-current">Current value <b>${current}</b></div>` : ""}
    <div class="im-block"><span class="im-lab">Overview</span><p>${info.what}</p></div>
    <div class="im-block"><span class="im-lab">Interpretation</span><p>${info.read}</p></div>
    ${(buy || sell) ? `<div class="im-block"><span class="im-lab">Signals</span>
      <div class="im-sigs">
        ${buy ? `<div class="im-sig long"><span class="im-sig-k">Long · Bullish</span>
          <span class="im-sig-t">${buy}</span></div>` : ""}
        ${sell ? `<div class="im-sig short"><span class="im-sig-k">Short · Bearish</span>
          <span class="im-sig-t">${sell}</span></div>` : ""}
      </div></div>` : ""}
    <div class="im-block"><span class="im-lab">Calculation</span>
      <code class="im-formula">${info.formula}</code></div>
    ${(!buy && !sell) ? `<div class="im-note">Context / confirmation tool — it sharpens
      the strategy's entries rather than issuing its own signal.</div>` : ""}
    ${info.proprietary ? `<div class="im-note">Built by ASTRA from raw trade and
      derivatives data — not available on standard charting sites.</div>` : ""}`;
  box.querySelector(".im-close").addEventListener("click", () => (modal.hidden = true));
  modal.hidden = false;
}

/* clicking a legend chip, a pane header, or the ⓘ button in the studies menu
   opens the card */
document.addEventListener("click", (e) => {
  const lg = e.target.closest(".lg[data-k], .pb-item[data-k], .std-info[data-k]");
  if (lg && lg.dataset.k) {
    e.preventDefault();
    e.stopPropagation();
    showIndicatorInfo(lg.dataset.k);
  }
});
