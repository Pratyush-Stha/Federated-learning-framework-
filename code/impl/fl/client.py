"""Minimal Flower NumPyClient for CIFAR-100 / FEMNIST / IoV.

Each client is given (model_factory, train_loader, val_loader, device).
The PQ encryption step is simulated by adding a configurable sleep that
follows the empirical S_pq distribution; this lets the queueing layer
observe realistic per-round service times without depending on a working
liboqs install.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import flwr as fl


class QRQClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        model_factory: Callable[[], torch.nn.Module],
        train_loader,
        val_loader,
        device: str = "cpu",
        local_epochs: int = 1,
        pq_overhead_sampler: Callable[[str], float] | None = None,
    ):
        self.cid = client_id
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.pq_overhead_sampler = pq_overhead_sampler or (lambda algo: 0.0)
        self.model = self.model_factory().to(self.device)

    # --- Flower API ----------------------------------------------------
    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        sd = self.model.state_dict()
        for k, v in zip(sd.keys(), parameters):
            sd[k] = torch.tensor(v)
        self.model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Snapshot the global parameters for the optional FedProx proximal term.
        global_snapshot = [
            torch.tensor(p, device=self.device).detach() for p in parameters
        ]
        mu = float(config.get("proximal_mu", 0.0) or 0.0)
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.local_epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                if mu > 0.0:
                    prox = 0.0
                    for p, g in zip(self.model.parameters(), global_snapshot):
                        prox = prox + ((p - g) ** 2).sum()
                    loss = loss + 0.5 * mu * prox
                loss.backward()
                opt.step()
        # Simulate PQ encryption overhead so the server-side queue sees it.
        time.sleep(self.pq_overhead_sampler(config.get("pq_scheme", "kyber512")))
        n = sum(len(b[0]) for b in self.train_loader)
        return self.get_parameters({}), n, {"client_id": self.cid}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = total = 0
        loss_total = 0.0
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss_total += loss_fn(logits, yb).item()
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
        return loss_total / max(total, 1), total, {"accuracy": correct / max(total, 1)}
