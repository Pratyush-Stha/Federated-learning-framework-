"""Multi-stage queue network from Section III-D of the paper.

Stage 1: wireless uplink         -> M/G/1
Stage 2: PQ encryption service   -> M/G/1
Stage 3: edge aggregation        -> M/M/c
Stage 4: blockchain validation   -> M/G/1

We provide both:
  - analytical mean delays (Pollaczek-Khinchine, Erlang-C) so the paper's
    closed-form claims can be verified, and
  - a SimPy-based simulator that returns the empirical distribution of
    end-to-end delay per round, used to compute hat-gamma in eq. (gamma).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np
import simpy


# ---------------------------------------------------------------------------
# Analytical formulas
# ---------------------------------------------------------------------------

def pk_mean_wait(lam: float, e_s: float, e_s2: float) -> float:
    """Pollaczek-Khinchine: mean waiting time in an M/G/1 queue."""
    rho = lam * e_s
    if rho >= 1.0:
        return float("inf")
    return lam * e_s2 / (2.0 * (1.0 - rho))


def erlang_c(c: int, rho_total: float) -> float:
    """Erlang-C blocking probability for M/M/c."""
    if rho_total >= c:
        return 1.0
    a = rho_total
    s = sum(a ** k / math.factorial(k) for k in range(c))
    p_block = (a ** c / math.factorial(c)) * (c / (c - a))
    return p_block / (s + p_block)


def mmc_mean_wait(lam: float, mu: float, c: int) -> float:
    """Mean waiting time in an M/M/c queue."""
    rho_total = lam / mu
    if rho_total >= c:
        return float("inf")
    pq = erlang_c(c, rho_total)
    return pq / (c * mu - lam)


# ---------------------------------------------------------------------------
# SimPy simulator
# ---------------------------------------------------------------------------

@dataclass
class StageSpec:
    name: str
    servers: int
    service_sampler: callable  # () -> float


@dataclass
class NetworkSpec:
    arrival_rate: float
    deadline: float
    stages: list[StageSpec]


def run_simulation(spec: NetworkSpec, n_arrivals: int = 5000, seed: int = 0):
    """Returns end-to-end delay samples and the deadline-success rate."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    resources = [
        simpy.Resource(env, capacity=s.servers) for s in spec.stages
    ]
    delays: list[float] = []

    def packet(idx: int):
        t0 = env.now
        for stage, res in zip(spec.stages, resources):
            with res.request() as req:
                yield req
                yield env.timeout(stage.service_sampler())
        delays.append(env.now - t0)

    def generator():
        for i in range(n_arrivals):
            yield env.timeout(rng.exponential(1.0 / spec.arrival_rate))
            env.process(packet(i))

    env.process(generator())
    env.run()
    on_time = sum(1 for d in delays if d <= spec.deadline)
    return delays, on_time / max(len(delays), 1)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _exp(mean: float):
    rng = np.random.default_rng()
    return lambda: float(rng.exponential(mean))


def _const(mean: float):
    return lambda: mean


if __name__ == "__main__":
    lam = 8.0  # 8 updates per second per edge

    spec = NetworkSpec(
        arrival_rate=lam,
        deadline=2.0,  # 2 seconds round deadline
        stages=[
            StageSpec("uplink",     servers=1, service_sampler=_exp(0.05)),
            StageSpec("pq_encrypt", servers=1, service_sampler=_exp(0.04)),
            StageSpec("agg_pool",   servers=4, service_sampler=_exp(0.06)),
            StageSpec("blockchain", servers=1, service_sampler=_const(0.10)),
        ],
    )

    delays, gamma = run_simulation(spec, n_arrivals=20000, seed=7)
    p95 = float(np.percentile(delays, 95))
    p99 = float(np.percentile(delays, 99))
    print(f"empirical mean   = {statistics.mean(delays):.4f} s")
    print(f"p95              = {p95:.4f} s")
    print(f"p99              = {p99:.4f} s")
    print(f"hat-gamma (T=2s) = {gamma:.4f}")

    # Analytical sanity checks for stage 1 (M/M/1, exp service mean=0.05):
    e_s, e_s2 = 0.05, 2 * 0.05 ** 2
    w_pk = pk_mean_wait(lam, e_s, e_s2)
    print(f"P-K wait (uplink) = {w_pk:.4f} s")
