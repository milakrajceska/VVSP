import numpy as np
import math
from collections import deque

np.random.seed(0)

DRINK = "drink"
FOOD = "food"
BOTH = "both"


def simulate_parallel(lambda_rate=5.0,
                      mu_barista=3.0,
                      mu_cook=3.0,
                      p_drink=0.5,
                      p_food=0.3,
                      p_both=0.2,
                      T=1000.0):

    t = 0.0
    next_arrival = np.random.exponential(1.0 / lambda_rate)

    # servers
    bar_busy = False
    cook_busy = False
    bar_end = math.inf
    cook_end = math.inf

    bar_queue = deque()
    cook_queue = deque()

    bar_client = None
    cook_client = None

    # tracking
    last_event = 0.0
    bar_busy_time = 0.0

    # clients state
    client_id = 0
    clients = {}

    def start_bar(client, time):
        nonlocal bar_busy, bar_end, bar_client
        bar_busy = True
        bar_client = client
        service_time = np.random.exponential(1.0 / mu_barista)
        bar_end = time + service_time

    def start_cook(client, time):
        nonlocal cook_busy, cook_end, cook_client
        cook_busy = True
        cook_client = client
        service_time = np.random.exponential(1.0 / mu_cook)
        cook_end = time + service_time

    while True:
        next_event = min(next_arrival, bar_end, cook_end)
        if next_event == math.inf:
            break

        t = next_event

        dt = t - last_event
        if bar_busy:
            bar_busy_time += dt
        last_event = t

        # ARRIVAL
        if t == next_arrival and t <= T:
            u = np.random.rand()

            if u < p_drink:
                ctype = DRINK
            elif u < p_drink + p_food:
                ctype = FOOD
            else:
                ctype = BOTH

            cid = client_id
            client_id += 1

            if ctype == DRINK:
                clients[cid] = {"bar_done": False, "cook_done": True}

                if not bar_busy:
                    start_bar(cid, t)
                else:
                    bar_queue.append(cid)

            elif ctype == FOOD:
                clients[cid] = {"bar_done": True, "cook_done": False}

                if not cook_busy:
                    start_cook(cid, t)
                else:
                    cook_queue.append(cid)

            else:  #both
                clients[cid] = {"bar_done": False, "cook_done": False}

                if not bar_busy:
                    start_bar(cid, t)
                else:
                    bar_queue.append(cid)

                if not cook_busy:
                    start_cook(cid, t)
                else:
                    cook_queue.append(cid)

            next_arrival = t + np.random.exponential(1.0 / lambda_rate) if t < T else math.inf

        # BAR FINISH
        elif t == bar_end:
            cid = bar_client
            bar_busy = False
            bar_end = math.inf
            bar_client = None

            clients[cid]["bar_done"] = True

            # check if BOTH finished
            if clients[cid]["bar_done"] and clients[cid]["cook_done"]:
                pass  # client leaves

            if bar_queue:
                nxt = bar_queue.popleft()
                start_bar(nxt, t)

        # COOK FINISH
        elif t == cook_end:
            cid = cook_client
            cook_busy = False
            cook_end = math.inf
            cook_client = None

            clients[cid]["cook_done"] = True

            if clients[cid]["bar_done"] and clients[cid]["cook_done"]:
                pass

            if cook_queue:
                nxt = cook_queue.popleft()
                start_cook(nxt, t)

        else:
            next_arrival = math.inf

        if (t >= T and not bar_busy and not cook_busy and
                not bar_queue and not cook_queue and next_arrival == math.inf):
            break

    total_time = max(T, t)
    rho_bar = bar_busy_time / total_time

    return rho_bar



def run_experiments():
    lambda_rate = 5.0
    mu_c_values = np.arange(0.5, 6.6, 0.5)

    p_drink = 0.5
    p_food = 0.3
    p_both = 0.2
    T = 1000.0

    # CASE 1
    print("\n=== CASE 1: lambda=5, mu_barista=3 ===")
    print("mu_cook\tUtil_barista")

    for mu_cook in mu_c_values:
        results = []
        for _ in range(30):
            rho = simulate_parallel(lambda_rate, 3.0, mu_cook,
                                    p_drink, p_food, p_both, T)
            results.append(rho)

        print(f"{mu_cook:.1f}\t{np.mean(results):.4f}")

    # CASE 2
    print("\n=== CASE 2: lambda=5, mu_barista=2 ===")
    print("mu_cook\tUtil_barista")

    for mu_cook in mu_c_values:
        results = []
        for _ in range(30):
            rho = simulate_parallel(lambda_rate, 2.0, mu_cook,
                                    p_drink, p_food, p_both, T)
            results.append(rho)

        print(f"{mu_cook:.1f}\t{np.mean(results):.4f}")


if __name__ == "__main__":
    run_experiments()