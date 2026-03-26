# Simulation-Based Modeling of a Coffee Bar as a Queueing System with Heterogeneous Servers

This project presents a discrete-event simulation of a coffee bar with heterogeneous servers (barista and cook) and probabilistic customer arrivals. It was presented at the “Math and its Application” seminar at the Faculty of Natural Sciences. The model captures realistic features such as parallel service for combined orders (customers ordering both drinks and food).

Project Goals
-Model a coffee bar as a queueing system with heterogeneous servers.
-Simulate parallel processing for customers requiring both drink and food.
-Analyze performance metrics: waiting time, queue length, server utilization.
-Perform sensitivity analysis on arrival rates and server capacities.
-Compare simulation results to M/M/1 theoretical approximations.

Key Features
-Discrete-event simulation of arrivals, departures, and server status.
-Three customer types:
   -Drink-only – served by the barista.
   -Food-only – served by the cook.
   -Combined orders – served in parallel by both servers.
-Tracks time-averaged metrics for queues and servers.
-Supports multiple independent simulation replications for statistical reliability.
-Includes sensitivity analysis to identify bottlenecks and system stability.

Insights & Findings
-The cook is typically the system bottleneck; increasing cook capacity can shift the bottleneck to the barista.
-Parallel service for combined orders reduces waiting times and improves overall throughput.
-Increasing arrival rates beyond a threshold leads to queue explosion and system instability.
-Simulation results match closely with M/M/1 analytical predictions for isolated servers, validating the model.

Practical Implications
-Managers should focus on improving cook efficiency to reduce bottlenecks.
-Parallel processing allows simultaneous preparation of drinks and food, which reduces waiting times for combined orders.
-Careful capacity planning is required to maintain system stability, especially under high demand.

Future Work
-Implement priority or load-balancing policies for different customer types.
-Model customer impatience (abandonment) for more realistic scenarios.
-Extend to multiple servers at each station (e.g., two baristas, two cooks).
-Calibrate simulation with real-world cafe data.
