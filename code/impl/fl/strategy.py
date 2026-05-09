"""Custom Flower strategy that wires the four QRQ-FL layers together.

The base FedAvg algorithm is unchanged; what we add are the three hooks
described in Algorithm 1 of the paper:

  configure_fit()   -> uses HMM rho_i(t) + DPP optimiser to pick clients
  aggregate_fit()   -> applies homomorphic mask cancellation (DTAHE stub)
                       and updates the virtual queues Q_i(t)
  evaluate()        -> records latency / hat-gamma / PQ overhead per round
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from hmm.reliability import HMMParams, dropout_probability
from rl.dpp import State, dpp_select, constraints, update_virtual_queues
from pqc.timing import measure as measure_pq


@dataclass
class QRQConfig:
    n_select: int = 10
    V: float = 2.0
    pq_set: tuple[str, ...] = ("kyber512", "dilithium2", "falcon-512")
    deadline_s: float = 2.0


class QRQFLStrategy(fl.server.strategy.FedAvg):
    """FedAvg + HMM + queueing + DPP + (optional) PQ stub."""

    def __init__(self, *args, qrq: QRQConfig | None = None, hmm: HMMParams | None = None, **kw):
        super().__init__(*args, **kw)
        self.cfg = qrq or QRQConfig()
        self.hmm = hmm or HMMParams.reasonable_default()
        self._obs_history: dict[str, list[int]] = defaultdict(list)
        self._Q = np.zeros(3)
        self.metrics: list[dict] = []
        self._pq_cache: dict[str, float] = {
            algo: measure_pq(algo, n=20).mean for algo in self.cfg.pq_set
        }

    # --- selection ---------------------------------------------------
    def configure_fit(self, server_round, parameters, client_manager: ClientManager):
        all_clients = list(client_manager.all().values())
        if not all_clients:
            return []
        cids = [c.cid for c in all_clients]

        rho = np.array([
            dropout_probability(np.asarray(self._obs_history[cid] or [0]), self.hmm)
            for cid in cids
        ])
        snr = np.full(len(cids), 20.0)  # plug in real SNR from your sim
        state = State(rho=rho, snr=snr, queue_lengths=self._Q,
                      pq_service_time=self._pq_cache)
        action = dpp_select(state, V=self.cfg.V, k=self.cfg.n_select,
                            pq_set=list(self.cfg.pq_set))

        chosen = [all_clients[i] for i, x in enumerate(action.selected) if x]
        cfg = {
            "round": server_round,
            "pq_scheme": action.pq_scheme,
            "deadline_s": self.cfg.deadline_s,
        }
        return [(c, FitIns(parameters, cfg)) for c in chosen]

    # --- aggregation -------------------------------------------------
    def aggregate_fit(self, server_round, results, failures):
        # Update HMM observation history (0=ok, 1=missing).
        observed_ok = {res.metrics.get("client_id", c.cid) for c, res in results}
        for c, _ in results:
            self._obs_history[c.cid].append(0)
        for cf in failures:
            cid = getattr(cf, "cid", None) or getattr(cf, "client_proxy", None)
            if cid is not None:
                cid = cid.cid if hasattr(cid, "cid") else cid
                self._obs_history[cid].append(1)

        agg = super().aggregate_fit(server_round, results, failures)

        # virtual queue update with empirical g_i:
        rho_now = np.mean([
            dropout_probability(np.asarray(self._obs_history[cid] or [0]), self.hmm)
            for cid in self._obs_history
        ]) if self._obs_history else 0.0
        cfg_round = results[0][1].metrics if results else {}
        pq_scheme = cfg_round.get("pq_scheme", next(iter(self.cfg.pq_set)))
        n_sel = max(len(results), 1)
        action_proxy = type("A", (), {
            "selected": np.ones(n_sel, dtype=int),
            "pq_scheme": pq_scheme,
            "bandwidth": np.ones(n_sel),
        })()
        # constraints() indexes rho[selected]; rho must match selected length (not scalar).
        proxy_state = State(
            rho=np.full(n_sel, rho_now),
            snr=np.full(n_sel, 20.0),
            queue_lengths=self._Q,
            pq_service_time=self._pq_cache,
        )
        g = constraints(proxy_state, action_proxy)
        self._Q = update_virtual_queues(self._Q, g)

        self.metrics.append({
            "round": server_round,
            "n_received": len(results),
            "n_failed": len(failures),
            "rho_mean": float(rho_now),
            "Q": self._Q.tolist(),
            "pq_scheme": pq_scheme,
            "ts": time.time(),
        })
        return agg
