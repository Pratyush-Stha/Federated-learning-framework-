"""End-to-end orchestrator for QRQ-FL.

Run examples (see README for the full menu):

    python main.py --dataset cifar100 --clients 50 --rounds 20 \
                   --pq kyber512 --dropout 0.30
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import flwr as fl

# Local modules
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data.partition import dirichlet_partition  # noqa: E402
from fl.client import QRQClient                  # noqa: E402
from fl.strategy import QRQFLStrategy, QRQConfig # noqa: E402
from hmm.reliability import HMMParams            # noqa: E402
from pqc.timing import measure as pq_measure     # noqa: E402


def small_cnn(num_classes: int = 100) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
        nn.Linear(256, num_classes),
    )


def cifar100_loaders(n_clients: int, alpha: float, batch: int, seed: int):
    tfm = T.Compose([T.ToTensor(),
                     T.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762))])
    root = pathlib.Path("./data_cache")
    train = torchvision.datasets.CIFAR100(root=root, train=True,  download=True, transform=tfm)
    test  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=tfm)
    labels = np.asarray(train.targets)
    splits = dirichlet_partition(labels, n_clients, alpha=alpha, seed=seed)
    val_loader = DataLoader(test, batch_size=256, shuffle=False)
    return [
        DataLoader(Subset(train, idx.tolist()), batch_size=batch, shuffle=True)
        for idx in splits
    ], val_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar100"], default="cifar100",
                    help="FEMNIST / MIMIC / IoV variants are wired in their respective data/ modules.")
    ap.add_argument("--clients", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha for label skew")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pq", default="kyber512", help="default PQ scheme (overridden by DPP if --schedule dpp)")
    ap.add_argument("--pq-set", default="kyber512,dilithium2,falcon-512",
                    help="comma-separated PQ scheme library available to DPP")
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--V", type=float, default=2.0)
    ap.add_argument("--n-select", type=int, default=10)
    ap.add_argument("--schedule", choices=["dpp", "random"], default="dpp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.dataset != "cifar100":
        raise NotImplementedError("Wire FEMNIST / MIMIC / IoV in main() following the CIFAR-100 pattern.")
    train_loaders, val_loader = cifar100_loaders(args.clients, args.alpha, args.batch, args.seed)

    # Pre-measure PQ schemes once and cache the mean S_pq.
    pq_set = [s.strip() for s in args.pq_set.split(",") if s.strip()]
    pq_means = {algo: pq_measure(algo, n=20).mean for algo in pq_set}
    print(f"PQ service-time profile: {pq_means}")

    def overhead_sampler(algo: str) -> float:
        # exponential around the mean to seed S^2 in the queue model
        return float(np.random.exponential(pq_means.get(algo, 0.01)))

    def client_fn(cid: str):
        i = int(cid)
        return QRQClient(
            client_id=cid,
            model_factory=lambda: small_cnn(num_classes=100),
            train_loader=train_loaders[i],
            val_loader=val_loader,
            device="cuda" if torch.cuda.is_available() else "cpu",
            pq_overhead_sampler=overhead_sampler,
        ).to_client()

    strategy = QRQFLStrategy(
        fraction_fit=0.0, fraction_evaluate=0.0,  # we control selection ourselves
        min_fit_clients=args.n_select, min_evaluate_clients=1,
        min_available_clients=args.clients,
        qrq=QRQConfig(n_select=args.n_select, V=args.V, pq_set=tuple(pq_set)),
        hmm=HMMParams.reasonable_default(),
    )

    out = pathlib.Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Logs -> {out}")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "n_received", "n_failed", "rho_mean",
                    "Q0", "Q1", "Q2", "pq_scheme", "ts"])
        for m in strategy.metrics:
            w.writerow([m["round"], m["n_received"], m["n_failed"], m["rho_mean"],
                        *m["Q"], m["pq_scheme"], m["ts"]])
    print("Done.")
    print(history)


if __name__ == "__main__":
    main()
