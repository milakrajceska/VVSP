import numpy as np
import heapq
import matplotlib.pyplot as plt
from math import factorial
from collections import deque

# za da se dobivaat isti brojki pri sekoe izvrshuvanje
np.random.seed(0)

def simulate_mm_m(lambda_rate, mu_rate, m, T, record_times=True):

    #pochetni vrednosti
    t = 0.0
    #vreme na prvo pristignuvanje
    next_arrival = np.random.exponential(1.0 / lambda_rate)
    busy_servers = []
    #red na klienti
    queue = deque()

    #vreminja na chekanje
    waiting_times = []
    #vreminja vo sistem (chekanje+usluga)
    system_times = []
    total_arrivals = 0
    area_queue = 0.0
    area_system = 0.0
    last_event_time = 0.0
    times = []
    qlens = []
    busy_time_acc = 0.0

    while (next_arrival <= T) or busy_servers or queue:
        next_departure = busy_servers[0] if busy_servers else float('inf')

        if next_arrival <= next_departure and next_arrival <= T:
            t = next_arrival
            dt = t - last_event_time
            area_queue += len(queue) * dt
            area_system += (len(queue) + len(busy_servers)) * dt
            busy_time_acc += dt * len(busy_servers)
            last_event_time = t
            total_arrivals += 1

            #ako ima sloboden server klientot pochnuva so usluga
            if len(busy_servers) < m:
                service_time = np.random.exponential(1.0 / mu_rate)
                heapq.heappush(busy_servers, t + service_time)
                waiting_times.append(0.0)
                system_times.append(service_time)
            else:
                #ako site serveri se zafateni odi vo redica
                queue.append(t)

            #sledno pristignuvanje
            next_arrival = t + np.random.exponential(1.0 / lambda_rate)

        else:
            #zavrshuvanje na usluga
            t = next_departure
            dt = t - last_event_time
            area_queue += len(queue) * dt
            area_system += (len(queue) + len(busy_servers)) * dt
            busy_time_acc += dt * len(busy_servers)
            last_event_time = t

            heapq.heappop(busy_servers)

            #ako ima red prviot klient vleguva za usluga
            if queue:
                arrival_time = queue.popleft()
                wait = t - arrival_time
                service_time = np.random.exponential(1.0 / mu_rate)
                heapq.heappush(busy_servers, t + service_time)
                waiting_times.append(wait)
                system_times.append(wait + service_time)

        if record_times:
            times.append(t)
            qlens.append(len(queue))

    if last_event_time < T:
        dt = T - last_event_time
        area_queue += len(queue) * dt
        area_system += (len(queue) + len(busy_servers)) * dt
        busy_time_acc += dt * len(busy_servers)
        last_event_time = T

    sim_duration = max(T, last_event_time)

    #Lq
    avg_queue = area_queue / sim_duration
    #L
    avg_system = area_system / sim_duration
    #Wq
    avg_wait = np.mean(waiting_times) if waiting_times else 0.0
    #W
    avg_system_time = np.mean(system_times) if system_times else 0.0
    #p
    utilization = busy_time_acc / (m * sim_duration)

    return {
        "lambda": lambda_rate,
        "mu": mu_rate,
        "m": m,
        "T": T,
        "total_arrivals": total_arrivals,
        "avg_queue_sim": avg_queue,
        "avg_system_sim": avg_system,
        "avg_wait_sim": avg_wait,
        "avg_system_time_sim": avg_system_time,
        "utilization_sim": utilization,
        "waiting_times": np.array(waiting_times),
        "system_times": np.array(system_times),
        "times": np.array(times),
        "qlens": np.array(qlens)
    }


def theoretical_mm_m(lambda_rate, mu_rate, m):
    rho = lambda_rate / mu_rate
    # optovaruvanje po server
    eps = rho / m                       

    # ako sistemot e nestabilen
    if eps >= 1.0:
        return {
            "p0": 0.0,
            "Pwait": 1.0,
            "Lq": float('inf'),
            "L": float('inf'),
            "Wq": float('inf'),
            "W": float('inf'),
            "rho": rho,
            "eps": eps
        }

    #verojatnost sistemot da e prazen
    sum_terms = sum((rho**n) / factorial(n) for n in range(m))
    last_term = (rho**m) / (factorial(m) * (1 - eps))
    p0 = 1.0 / (sum_terms + last_term)

    #verojatnosta deka klient ke cheka
    Pwait = last_term * p0

    Lq = Pwait * (rho / m) / (1 - eps)
    L = Lq + rho
    Wq = Lq / lambda_rate
    W = Wq + 1.0 / mu_rate

    return {
        "p0": p0,
        "Pwait": Pwait,
        "Lq": Lq,
        "L": L,
        "Wq": Wq,
        "W": W,
        "rho": rho,
        "eps": eps
    }

if __name__ == "__main__":
    lambda_rate = 4.0   # λ=4 klienti/h
    mu_rate = 2.0       # μ = 2 uslugi/h po server
    T = 1000.0          # vremetraenje na simulacija
    m_values = [1, 2, 3, 4, 5]  #za razl br na serveri

    #listi za rezultati
    sim_Wq = []
    theor_Wq = []
    sim_Lq = []
    theor_Lq = []

    for m in m_values:

        sim = simulate_mm_m(lambda_rate, mu_rate, m, T)
        sim_Wq.append(sim['avg_wait_sim'])
        sim_Lq.append(sim['avg_queue_sim'])

        theor = theoretical_mm_m(lambda_rate, mu_rate, m)
        theor_Wq.append(theor['Wq'])
        theor_Lq.append(theor['Lq'])

        #pechatenje za konkretno m
        print(f"\n Rezultati za m = {m} ")
        print(f"Simulacija: Wq={sim['avg_wait_sim']:.3f}, Lq={sim['avg_queue_sim']:.3f}")
        print(f"Teorija : Wq={theor['Wq']:.3f}, Lq={theor['Lq']:.3f}")

    # Wq vs m
    plt.figure(figsize=(8,4))
    plt.plot(m_values, sim_Wq, 'o-', label='Simulacija')
    plt.plot(m_values, theor_Wq, 's--', label='Teorija')
    plt.xlabel('Broj na serveri m')
    plt.ylabel('Prosechno vreme na chekanje Wq')
    plt.title('Wq vs broj na serveri')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Lq vs m
    plt.figure(figsize=(8,4))
    plt.plot(m_values, sim_Lq, 'o-', label='Simulacija')
    plt.plot(m_values, theor_Lq, 's--', label='Teorija')
    plt.xlabel('Broj na serveri m')
    plt.ylabel('Prosechen broj na klienti vo red Lq')
    plt.title('Lq vs broj na serveri')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

