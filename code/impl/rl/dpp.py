"""Drift-Plus-Penalty per-slot action selector (eq. (dpp_action) of the paper).

Solves
    a*_t in argmax_a  V * r(s,a)  -  sum_i Q_i(t) g_i(s,a)
over a finite candidate set generated from the available clients and the
PQ-scheme library. For very large action spaces, swap this for the PPO
agent in stable_baselines3 (see fl/strategy.py for the hook).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    rho: np.ndarray              # per-client dropout posterior, shape (N,)
    snr: np.ndarray              # per-client SNR proxy,         shape (N,)
    queue_lengths: np.ndarray    # virtual queue backlogs Q_i,   shape (M,)
    pq_service_time: dict[str, float]   # {algo: E[S_pq]}


@dataclass
class Action:
    selected: np.ndarray         # binary, shape (N,)
    pq_scheme: str
    bandwidth: np.ndarray        # per-selected-client share, shape (N,)


def reward(state: State, action: Action, *, w_acc=1.0, w_lat=0.5, w_eng=0.1) -> float:
    """Single-step reward used by both the DPP optimiser and PPO."""
    sel = action.selected.astype(bool)
    if not sel.any():
        return -1.0  # no participation -> heavy penalty
    expected_acc_gain = float(w_acc * (1.0 - state.rho[sel]).mean())
    expected_latency  = state.pq_service_time[action.pq_scheme]
    expected_energy   = float(sel.mean())
    return expected_acc_gain - w_lat * expected_latency - w_eng * expected_energy


def constraints(state: State, action: Action) -> np.ndarray:
    """g_i(s,a) for each virtual queue. Negative is good (strictly feasible)."""
    sel = action.selected.astype(bool)
    util = sel.mean()
    pq_cost = state.pq_service_time[action.pq_scheme] - 0.10  # SLA: 100 ms
    drop_cost = float(state.rho[sel].mean()) - 0.30 if sel.any() else 0.0
    return np.array([util - 0.6, pq_cost, drop_cost])


def candidate_actions(state: State, k: int, pq_set: list[str], samples: int = 256, seed: int = 0):
    """Generate `samples` random feasible actions: pick top-2k by reliability,
    sample k of them, and try every PQ scheme."""
    rng = np.random.default_rng(seed)
    N = len(state.rho)
    pool = np.argsort(state.rho)[: min(2 * k, N)]
    for _ in range(samples):
        chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
        sel = np.zeros(N, dtype=int)
        sel[chosen] = 1
        bw = np.zeros(N)
        bw[chosen] = 1.0 / max(len(chosen), 1)
        for pq in pq_set:
            yield Action(selected=sel, pq_scheme=pq, bandwidth=bw)


def dpp_select(state: State, V: float, k: int, pq_set: list[str], samples: int = 256) -> Action:
    """Per-slot DPP optimiser."""
    best, best_score = None, -math.inf
    for a in candidate_actions(state, k, pq_set, samples):
        r = reward(state, a)
        g = constraints(state, a)
        score = V * r - float(state.queue_lengths @ np.maximum(g, 0))
        if score > best_score:
            best_score, best = score, a
    assert best is not None
    return best


def update_virtual_queues(Q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Eq. (virtual_queue): Q_i(t+1) = max(Q_i(t) + g_i, 0)."""
    return np.maximum(Q + g, 0.0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 30
    state = State(
        rho=rng.beta(2, 5, size=N),
        snr=rng.uniform(5, 30, size=N),
        queue_lengths=np.zeros(3),
        pq_service_time={"kyber512": 0.01, "dilithium2": 0.08, "falcon-512": 1.10},
    )
    Q = state.queue_lengths.copy()
    for t in range(20):
        a = dpp_select(state, V=2.0, k=10, pq_set=list(state.pq_service_time))
        g = constraints(state, a)
        Q = update_virtual_queues(Q, g)
        if t % 5 == 0:
            print(f"t={t:02d}  pq={a.pq_scheme:<11}  picked={a.selected.sum():2d}  Q={np.round(Q,3)}")
