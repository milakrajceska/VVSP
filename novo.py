import numpy as np
import math
from collections import deque

DRINK = "drink"
FOOD  = "food"
BOTH  = "both"


def simulate_parallel(lambda_rate=5.0,
                      mu_barista=6.0,
                      mu_cook=3.0,
                      p_drink=0.5,
                      p_food=0.3,
                      p_both=0.2,
                      T=1000.0,
                      seed=0):
    """
    Returns a dict with:
        rho_bar, rho_cook,
        Lq_bar, Lq_cook,
        Wq_all, Wq_both,
        n_departed
    """
    rng = np.random.default_rng(seed)

    t          = 0.0
    last_event = 0.0

    # ── servers ────────────────────────────────────────────────────────────
    bar_busy  = False;  bar_end  = math.inf;  bar_client  = None
    cook_busy = False;  cook_end = math.inf;  cook_client = None

    bar_queue  = deque()
    cook_queue = deque()

    # ── accumulators ───────────────────────────────────────────────────────
    bar_busy_time  = 0.0
    cook_busy_time = 0.0
    Lq_bar_acc     = 0.0   # ∫ |bar_queue| dt
    Lq_cook_acc    = 0.0   # ∫ |cook_queue| dt

    # ── per-customer records ───────────────────────────────────────────────
    # each entry: {type, arrive, bar_wait_start, cook_wait_start,
    #              bar_done, cook_done, wq}
    clients    = {}
    client_id  = 0
    departed   = []        # list of Wq per departed customer
    departed_both = []     # Wq for BOTH customers only

    next_arrival = rng.exponential(1.0 / lambda_rate)

    # ── helpers ────────────────────────────────────────────────────────────
    def start_bar(cid, now):
        nonlocal bar_busy, bar_end, bar_client
        bar_busy   = True
        bar_client = cid
        bar_end    = now + rng.exponential(1.0 / mu_barista)
        # if this client was waiting, accumulate wait
        c = clients[cid]
        if c["bar_wait_start"] is not None:
            c["wq"] += now - c["bar_wait_start"]
            c["bar_wait_start"] = None

    def start_cook(cid, now):
        nonlocal cook_busy, cook_end, cook_client
        cook_busy   = True
        cook_client = cid
        cook_end    = now + rng.exponential(1.0 / mu_cook)
        c = clients[cid]
        if c["cook_wait_start"] is not None:
            c["wq"] += now - c["cook_wait_start"]
            c["cook_wait_start"] = None

    def try_depart(cid, now):
        c = clients[cid]
        if c["bar_done"] and c["cook_done"]:
            departed.append(c["wq"])
            if c["type"] == BOTH:
                departed_both.append(c["wq"])

    # ── main loop ──────────────────────────────────────────────────────────
    while True:
        next_event = min(next_arrival, bar_end, cook_end)
        if next_event == math.inf:
            break

        t  = next_event
        dt = t - last_event

        # accumulate time-weighted stats
        bar_busy_time  += dt * bar_busy
        cook_busy_time += dt * cook_busy
        Lq_bar_acc     += dt * len(bar_queue)
        Lq_cook_acc    += dt * len(cook_queue)

        last_event = t

        # ── ARRIVAL ────────────────────────────────────────────────────────
        if t == next_arrival and t <= T:
            u = rng.random()
            if   u < p_drink:            ctype = DRINK
            elif u < p_drink + p_food:   ctype = FOOD
            else:                        ctype = BOTH

            cid = client_id
            client_id += 1
            clients[cid] = {
                "type": ctype, "arrive": t, "wq": 0.0,
                "bar_done":  (ctype == FOOD),
                "cook_done": (ctype == DRINK),
                "bar_wait_start":  None,
                "cook_wait_start": None,
            }

            if ctype in (DRINK, BOTH):
                if not bar_busy:
                    start_bar(cid, t)
                else:
                    clients[cid]["bar_wait_start"] = t
                    bar_queue.append(cid)

            if ctype in (FOOD, BOTH):
                if not cook_busy:
                    start_cook(cid, t)
                else:
                    clients[cid]["cook_wait_start"] = t
                    cook_queue.append(cid)

            next_arrival = (t + rng.exponential(1.0 / lambda_rate)
                            if t < T else math.inf)

        # ── BAR FINISH ─────────────────────────────────────────────────────
        elif t == bar_end:
            cid       = bar_client
            bar_busy  = False
            bar_end   = math.inf
            bar_client = None
            clients[cid]["bar_done"] = True
            try_depart(cid, t)

            if bar_queue:
                start_bar(bar_queue.popleft(), t)

        # ── COOK FINISH ────────────────────────────────────────────────────
        elif t == cook_end:
            cid        = cook_client
            cook_busy  = False
            cook_end   = math.inf
            cook_client = None
            clients[cid]["cook_done"] = True
            try_depart(cid, t)

            if cook_queue:
                start_cook(cook_queue.popleft(), t)

        else:
            next_arrival = math.inf

        if (t >= T and not bar_busy and not cook_busy
                and not bar_queue and not cook_queue
                and next_arrival == math.inf):
            break

    total_time = max(T, t)

    Wq_all  = np.mean(departed)      if departed      else 0.0
    Wq_both = np.mean(departed_both) if departed_both else 0.0

    return {
        "rho_bar":  bar_busy_time  / total_time,
        "rho_cook": cook_busy_time / total_time,
        "Lq_bar":   Lq_bar_acc     / total_time,
        "Lq_cook":  Lq_cook_acc    / total_time,
        "Wq_all":   Wq_all,
        "Wq_both":  Wq_both,
        "n_departed": len(departed),
    }


def ci95(data):
    """Return (mean, half-width) for 95 % CI."""
    data = np.array(data)
    mean = np.mean(data)
    hw   = 1.96 * np.std(data, ddof=1) / math.sqrt(len(data))
    return mean, hw


def replicate(n=30, **kwargs):
    """Run n replications with seeds 0..n-1 and return list of result dicts."""
    return [simulate_parallel(seed=i, **kwargs) for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
#  1.  M/M/1 VALIDATION  (p_drink=1, λ=4, μ_b=5)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("M/M/1 VALIDATION  (p_drink=1, λ=4, μ_barista=5)")
print("=" * 60)
theory_Lq = 0.8**2 / (1 - 0.8)   # ρ=0.8 → Lq=3.2
results_mm1 = replicate(
    n=30,
    lambda_rate=4.0, mu_barista=5.0, mu_cook=1.0,   # cook irrelevant
    p_drink=1.0, p_food=0.0, p_both=0.0,
    T=1000.0
)
Lq_vals = [r["Lq_bar"] for r in results_mm1]
m, hw    = ci95(Lq_vals)
print(f"  Theoretical Lq = {theory_Lq:.4f}")
print(f"  Simulated  Lq  = {m:.4f}  ±  {hw:.4f}  (95 % CI)  [{m-hw:.4f}, {m+hw:.4f}]")


# ═══════════════════════════════════════════════════════════════════════════
#  2.  BASE SCENARIO – barista and cook Lq with CI  (for Section VI.C)
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("BASE SCENARIO  (λ=5, μ_b=6, μ_c=3, p_drink=0.5, p_food=0.3, p_both=0.2)")
print("=" * 60)
results_base = replicate(
    n=30,
    lambda_rate=5.0, mu_barista=6.0, mu_cook=3.0,
    p_drink=0.5, p_food=0.3, p_both=0.2,
    T=1000.0
)
for key, label, theory in [
    ("Lq_bar",  "Lq_barista", 0.583**2 / (1 - 0.583)),
    ("Lq_cook", "Lq_cook",    0.833**2 / (1 - 0.833)),
    ("rho_bar", "ρ_barista",  0.583),
    ("rho_cook","ρ_cook",     0.833),
    ("Wq_all",  "Wq_all",     None),
    ("Wq_both", "Wq_both",    None),
]:
    vals     = [r[key] for r in results_base]
    m, hw    = ci95(vals)
    theory_s = f"  theory={theory:.4f}" if theory is not None else ""
    print(f"  {label:12s}  sim={m:.4f} ± {hw:.4f}  [{m-hw:.4f}, {m+hw:.4f}]{theory_s}")


# ═══════════════════════════════════════════════════════════════════════════
#  3.  SENSITIVITY – cook service rate  (3 μ_b values)
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SENSITIVITY: cook service rate  (λ=5, three μ_barista values)")
print("=" * 60)
mu_c_values = np.arange(0.5, 6.6, 0.5)

for mu_b in [2.0, 3.0, 4.0]:
    print(f"\n  --- μ_barista = {mu_b} ---")
    print(f"  {'mu_cook':>8}  {'rho_bar':>10}  {'CI_hw':>8}")
    for mu_c in mu_c_values:
        res   = replicate(n=30, lambda_rate=5.0, mu_barista=mu_b, mu_cook=mu_c,
                          p_drink=0.5, p_food=0.3, p_both=0.2, T=1000.0)
        vals  = [r["rho_bar"] for r in res]
        m, hw = ci95(vals)
        print(f"  {mu_c:8.1f}  {m:10.4f}  {hw:8.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  4.  SENSITIVITY – arrival rate  (with CIs on all reported columns)
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SENSITIVITY: arrival rate  (μ_b=6, μ_c=3)")
print("=" * 60)
print(f"  {'lambda':>8}  {'Lq_cook':>10}  {'CI_hw':>8}  {'Wq_both':>10}  {'CI_hw':>8}  {'rho_cook':>10}  {'CI_hw':>8}")
for lam in [4.0, 5.0, 6.0, 7.0, 8.0]:
    res = replicate(n=30, lambda_rate=lam, mu_barista=6.0, mu_cook=3.0,
                    p_drink=0.5, p_food=0.3, p_both=0.2, T=1000.0)
    lq_m,  lq_hw  = ci95([r["Lq_cook"] for r in res])
    wq_m,  wq_hw  = ci95([r["Wq_both"] for r in res])
    rho_m, rho_hw = ci95([r["rho_cook"] for r in res])
    print(f"  {lam:8.1f}  {lq_m:10.2f}  {lq_hw:8.2f}  {wq_m:10.2f}  {wq_hw:8.2f}  {rho_m:10.4f}  {rho_hw:8.4f}")