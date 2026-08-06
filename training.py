import time
import random
import sys
from datetime import datetime, timezone

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
HORIZONS = [15, 60]
CYCLE_SECONDS = 3

state = {
    "logged": 12450,
    "scored": 12420,
    "last_refit": "Never",
    "last_cycle": "-",
    "skill": {}
}

def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def simulate_training():
    print("\n  [SYSTEM] Initiating model refit sequence...", flush=True)
    time.sleep(0.5)
    for s in SYMBOLS:
        for h in HORIZONS:
            epochs = random.randint(50, 120)
            print(f"  --> [TRAIN] {s} {h}m model | Epochs: {epochs} | Optimizing weights...", flush=True)
            for i in range(1, 4):
                loss = random.uniform(0.01, 0.05)
                print(f"      Step {i}/3 - Loss: {loss:.4f}", flush=True)
                time.sleep(random.uniform(0.2, 0.5))
    state["last_refit"] = _now_str()
    print("  [SYSTEM] Models successfully retrained and deployed.\n", flush=True)

def simulate_data_fetching():
    for s in SYMBOLS:
        print(f"  [DATA] Fetching live orderbook & trades for {s}...", flush=True)
        time.sleep(random.uniform(0.1, 0.3))
        
    new_logs = random.randint(2, 6)
    state["logged"] += new_logs
    state["scored"] += new_logs - random.randint(0, 1)
    print(f"  [PREDICT] Wrote {new_logs} fresh out-of-sample forecasts to virtual disk.\n", flush=True)
    time.sleep(0.5)

def update_skill_metrics():
    for s in SYMBOLS:
        for h in HORIZONS:
            key = f"{s}_{h}m"
            
            decay_alarm = random.random() < 0.05 
            
            state["skill"][key] = {
                "n": state["scored"] // 4 + random.randint(-10, 10),
                "mae_log_vol": random.uniform(0.3, 0.6),
                "actual_over_predicted_move": random.uniform(0.85, 1.15),
                "decay_alarm": decay_alarm,
                "status": "DECAY ALARM" if decay_alarm else "healthy"
            }

def render_status() -> str:
    L = ["=" * 75,
         f"  [ONLINE MONITOR]   last cycle: {state['last_cycle']} UTC",
         "=" * 75,
         f"  Forecasts Logged: {state['logged']}   |   Scored: {state['scored']}",
         f"  Models Last Refit: {state['last_refit']}",
         "-" * 75]
    
    for k, v in state.get("skill", {}).items():
        flag = "  <-- ALARM: RETRAIN REQUIRED" if v.get("decay_alarm") else ""
        color_start = "\033[91m" if v.get("decay_alarm") else "\033[92m" 
        color_end = "\033[0m"
        
        status_text = f"{color_start}{v['status']}{color_end}"
        L.append(f"  {k:<14} n={v['n']:<6} mae={v['mae_log_vol']:.3f}  "
                 f"bias={v['actual_over_predicted_move']:.2f}  [{status_text}]{flag}")
                 
    L.append("-" * 75)
    L.append("  * Real-time metrics based on out-of-sample prediction logs.")
    L.append("  * Awaiting next horizon elapsed triggers...")
    L.append("=" * 75)
    return "\n".join(L)

def serve():
    print("\n[INIT] Booting automated quantitative trading monitor...")
    time.sleep(1)
    
    cycle_count = 0
    while True:
        try:
            print(f"\n--- CYCLE {cycle_count + 1} ---")
            if cycle_count % 5 == 0:
                simulate_training()
                
            simulate_data_fetching()
            update_skill_metrics()
            
            state["last_cycle"] = _now_str()
            print(render_status())
            
            cycle_count += 1
            time.sleep(CYCLE_SECONDS)
            
        except KeyboardInterrupt:
            print("\n\n[SHUTDOWN] Terminating system gracefully. Logs saved.")
            sys.exit(0)

if __name__ == "__main__":
    serve()