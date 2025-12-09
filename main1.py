import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt

DRINK = "drink"
FOOD = "food"
BOTH = "both"

def simulate_cafe_bar(lambda_rate=5.0,
                      mu_barista=6.0,
                      mu_cook=5.0,
                      p_drink=0.5,
                      p_food=0.3,
                      p_both=0.2,
                      T=1000.0,
                      record_process=False):

    #za isti rez sekogash
    np.random.seed(0)

    t = 0.0
    #prvo pristignuvanje
    next_arrival = np.random.exponential(1.0 / lambda_rate)

    bar_busy = False
    cook_busy = False
    bar_end = math.inf
    cook_end = math.inf
    bar_queue = deque()
    cook_queue = deque()
    bar_client = None
    cook_client = None

    #vreme vo redica Wq
    waiting_times = {DRINK: [], FOOD: [], BOTH: []}
    #vreme vo sistem W
    system_times = {DRINK: [], FOOD: [], BOTH: []}

    total_arrivals = 0
    type_counts = {DRINK: 0, FOOD: 0, BOTH: 0}

    last_event = 0.0
    area_bar_q = 0.0
    area_cook_q = 0.0
    area_system = 0.0

    if record_process:
        times = []
        bar_q_len = []
        cook_q_len = []

    def start_service(server, client, current_time):
        nonlocal bar_busy, cook_busy, bar_end, cook_end, bar_client, cook_client

        #ako bil vo red, -chekanjeto
        if client.get("queue_enter_time") is not None:
            client["waiting_accum"] += current_time - client["queue_enter_time"]
            client["queue_enter_time"] = None

        #prv pat vlegva na server
        if client["first_service_start"] is None:
            client["first_service_start"] = current_time

        if server == "bar":
            bar_busy = True
            bar_client = client
            service_time = np.random.exponential(1.0 / mu_barista)
            bar_end = current_time + service_time
        else:
            cook_busy = True
            cook_client = client
            service_time = np.random.exponential(1.0 / mu_cook)
            cook_end = current_time + service_time

    while True:
        next_event_time = min(next_arrival, bar_end, cook_end)
        if next_event_time == math.inf:
            break

        t = next_event_time

        #Lq i L
        dt = t - last_event #vreme od posledniot nastan do segashniot
        #Lq
        area_bar_q += len(bar_queue) * dt
        area_cook_q += len(cook_queue) * dt

        #L
        total_in_system = len(bar_queue) + len(cook_queue)
        if bar_busy:
            total_in_system += 1
        if cook_busy:
            total_in_system += 1
        area_system += total_in_system * dt

        last_event = t

        #pristignuvanje
        if t == next_arrival and t <= T:
            total_arrivals += 1

            #odreduvanje tip klient
            u = np.random.rand()
            if u < p_drink:
                ctype = DRINK
            elif u < p_drink + p_food:
                ctype = FOOD
            else:
                ctype = BOTH
            type_counts[ctype] += 1

            if ctype == BOTH:
                #kaj sho ima pomalce odi
                load_bar = len(bar_queue) + (1 if bar_busy else 0)
                load_cook = len(cook_queue) + (1 if cook_busy else 0)
                if load_bar <= load_cook:
                    stages = ["bar", "cook"]
                else:
                    stages = ["cook", "bar"]
            elif ctype == DRINK:
                stages = ["bar"]
            else:  #food
                stages = ["cook"]

            client = {
                "type": ctype,
                "arrival": t,
                "waiting_accum": 0.0,
                "first_service_start": None,
                "queue_enter_time": None,
                "remaining": stages
            }

            first_stage = client["remaining"].pop(0)
            if first_stage == "bar":
                if not bar_busy:
                    start_service("bar", client, t)
                else:
                    client["queue_enter_time"] = t
                    bar_queue.append(client)
            else:  #first_stage==cook
                if not cook_busy:
                    start_service("cook", client, t)
                else:
                    client["queue_enter_time"] = t
                    cook_queue.append(client)

            # закажи ново пристигнување
            if t < T:
                next_arrival = t + np.random.exponential(1.0 / lambda_rate)
            else:
                next_arrival = math.inf

        #zavrshuvanje usluga kaj barista
        elif t == bar_end:
            client = bar_client
            bar_busy = False
            bar_end = math.inf
            bar_client = None

            if client["remaining"]:
                next_stage = client["remaining"].pop(0)
                if next_stage == "cook":
                    if not cook_busy:
                        start_service("cook", client, t)
                    else:
                        client["queue_enter_time"] = t
                        cook_queue.append(client)
                else:
                    raise RuntimeError("ne e cook")
            else:
                #klientot go napushta sistemot
                waiting_times[client["type"]].append(client["waiting_accum"])
                system_times[client["type"]].append(t - client["arrival"])

            #zemi nov klient od redot kaj barista
            if bar_queue:
                nxt = bar_queue.popleft()
                start_service("bar", nxt, t)

        #zavrshuvanje na usluga kaj gotvach
        elif t == cook_end:
            client = cook_client
            cook_busy = False
            cook_end = math.inf
            cook_client = None

            if client["remaining"]:
                next_stage = client["remaining"].pop(0)
                if next_stage == "bar":
                    if not bar_busy:
                        start_service("bar", client, t)
                    else:
                        client["queue_enter_time"] = t
                        bar_queue.append(client)
                else:
                    raise RuntimeError("ne e bar")
            else:
                waiting_times[client["type"]].append(client["waiting_accum"])
                system_times[client["type"]].append(t - client["arrival"])

            if cook_queue:
                nxt = cook_queue.popleft()
                start_service("cook", nxt, t)

        else:
            #t>T
            next_arrival = math.inf

        if record_process:
            times.append(t)
            bar_q_len.append(len(bar_queue))
            cook_q_len.append(len(cook_queue))

        #uslov za zavrshuvanje
        if (t >= T and
            not bar_busy and not cook_busy and
            not bar_queue and not cook_queue and
            next_arrival == math.inf):
            break

    sim_time = max(T, last_event)
    avg_bar_q = area_bar_q / sim_time
    avg_cook_q = area_cook_q / sim_time
    avg_L = area_system / sim_time

    #za da ne delam so 0
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    results = {
        "lambda": lambda_rate,
        "mu_barista": mu_barista,
        "mu_cook": mu_cook,
        "p_drink": p_drink,
        "p_food": p_food,
        "p_both": p_both,
        "T": T,
        "total_arrivals": total_arrivals,
        "type_counts": type_counts,
        "avg_bar_queue": avg_bar_q,
        "avg_cook_queue": avg_cook_q,
        "avg_L": avg_L,
        "avg_wait_DRINK": safe_mean(waiting_times[DRINK]),
        "avg_wait_FOOD": safe_mean(waiting_times[FOOD]),
        "avg_wait_BOTH": safe_mean(waiting_times[BOTH]),
        "avg_system_DRINK": safe_mean(system_times[DRINK]),
        "avg_system_FOOD": safe_mean(system_times[FOOD]),
        "avg_system_BOTH": safe_mean(system_times[BOTH]),
        "waiting_times": waiting_times,
        "system_times": system_times
    }

    if record_process:
        results["times"] = times
        results["bar_q_len"] = bar_q_len
        results["cook_q_len"] = cook_q_len

    return results


#teoretski
def theoretical_mm1(lambda_rate, mu_rate):
    if lambda_rate >= mu_rate:
        return {
            "stable": False,
            "rho": lambda_rate / mu_rate if mu_rate > 0 else math.inf,
            "Lq": math.inf,
            "Wq": math.inf
        }

    rho = lambda_rate / mu_rate
    L = lambda_rate / (mu_rate - lambda_rate)
    W = 1.0 / (mu_rate - lambda_rate)
    Lq = (lambda_rate ** 2) / (mu_rate * (mu_rate - lambda_rate))
    Wq = Lq / lambda_rate

    return {
        "stable": True,
        "rho": rho,
        "L": L,
        "W": W,
        "Lq": Lq,
        "Wq": Wq
    }


if __name__ == "__main__":
    lambda_rate = 5.0
    mu_barista = 6.0
    mu_cook = 3.0

    p_drink = 0.5
    p_food = 0.3
    p_both = 0.2

    T = 1000.0

    sim = simulate_cafe_bar(lambda_rate, mu_barista, mu_cook,
                            p_drink, p_food, p_both, T,
                            record_process=True)

    #
    lambda_bar = lambda_rate * (p_drink + p_both)
    lambda_cook = lambda_rate * (p_food + p_both)

    theor_bar = theoretical_mm1(lambda_bar, mu_barista)
    theor_cook = theoretical_mm1(lambda_cook, mu_cook)

    print("\n=== Parametri na modelot ===")
    print(f"λ = {lambda_rate}, μ_bar = {mu_barista}, μ_cook = {mu_cook}")
    print(f"p(drink)={p_drink}, p(food)={p_food}, p(both)={p_both}")
    print(f"Vkupno pristignuvanja: {sim['total_arrivals']}")
    print(f"Po tip: {sim['type_counts']}")

    print("\n=== Prosechni dolzhini na redovi(simulacija) ===")
    print(f"Barista Lq_bar (sim) = {sim['avg_bar_queue']:.3f}")
    print(f"Gotvach  Lq_cook (sim) = {sim['avg_cook_queue']:.3f}")
    print(f"Vkupno L (sim) = {sim['avg_L']:.3f}")

    print("\n=== Teoretski ===")
    print(f"λ_bar = {lambda_bar:.2f}, λ_cook = {lambda_cook:.2f}")
    print(f"Lq_bar (theory)  = {theor_bar['Lq']:.3f}, Wq_bar = {theor_bar['Wq']:.3f}")
    print(f"Lq_cook (theory) = {theor_cook['Lq']:.3f}, Wq_cook = {theor_cook['Wq']:.3f}")

    print("\n=== Prosechno vreme na chekanje po tip klient(simulacija) ===")
    print(f"Wq DRINK = {sim['avg_wait_DRINK']:.3f}")
    print(f"Wq FOOD  = {sim['avg_wait_FOOD']:.3f}")
    print(f"Wq BOTH  = {sim['avg_wait_BOTH']:.3f}")

    print("\n=== Prosechno vreme na prestoj vo sistem ===")
    print(f"W DRINK = {sim['avg_system_DRINK']:.3f}")
    print(f"W FOOD  = {sim['avg_system_FOOD']:.3f}")
    print(f"W BOTH  = {sim['avg_system_BOTH']:.3f}")

    times = sim["times"]
    bar_q = sim["bar_q_len"]
    cook_q = sim["cook_q_len"]

    plt.figure(figsize=(9, 4))
    plt.plot(times, bar_q, label="Red kaj barista")
    plt.plot(times, cook_q, label="Red kaj gotvach", linestyle="--")
    plt.xlabel("Vreme")
    plt.ylabel("Broj na klienti vo red")
    plt.title("Dolzhina na red vo tek na vremeto")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    labels = ["Barista", "Cook"]
    sim_values = [sim["avg_bar_queue"], sim["avg_cook_queue"]]
    theory_values = [theor_bar["Lq"], theor_cook["Lq"]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, sim_values, width, label="Simulacija", color="skyblue")
    plt.bar(x + width / 2, theory_values, width, label="Teorija", color="orange")

    plt.xticks(x, labels)
    plt.ylabel("Lq")
    plt.title("Teorija vs Simulacija na Lq")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    rho_bar = lambda_bar / mu_barista
    rho_cook = lambda_cook / mu_cook

    plt.figure(figsize=(6, 4))
    plt.bar(["Barista", "Cook"], [rho_bar, rho_cook], color=["lightblue", "salmon"])
    plt.title("Optovarenost na servetrite (ρ)")
    plt.ylabel("ρ")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    wt = sim["waiting_times"]

    plt.figure(figsize=(10, 4))
    plt.hist(wt[DRINK], bins=30, alpha=0.6, label="DRINK")
    plt.hist(wt[FOOD], bins=30, alpha=0.6, label="FOOD")
    plt.hist(wt[BOTH], bins=30, alpha=0.6, label="BOTH")

    plt.title("Histogram na vremeto na chekanje Wq po tip klient")
    plt.xlabel("Vreme na chekanje")
    plt.ylabel("Frekvencija")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(["DRINK", "FOOD", "BOTH"],
            [sim["avg_wait_DRINK"], sim["avg_wait_FOOD"], sim["avg_wait_BOTH"]],
            color=["skyblue", "salmon", "lightgreen"])

    plt.title("Prosecno vreme na cekanje Wq po tip klient")
    plt.ylabel("Wq")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()





