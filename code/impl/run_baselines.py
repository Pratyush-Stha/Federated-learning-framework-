"""Run the full publication-baseline grid under a single Flower protocol.

Methods (all share dataset, partition, model, seed, rounds, batch size):

  * fedavg            -- random sampling, no PQ overhead
  * fedavg_kyber      -- random sampling, fixed Kyber PQ overhead
  * fedprox           -- random sampling, FedProx proximal mu>0
  * qrqfl             -- HMM + DPP + adaptive PQ scheme (full QRQ-FL)
  * qrqfl_no_hmm      -- DPP active, HMM disabled (uniform reliability)
  * qrqfl_no_dpp      -- HMM active, DPP disabled (top-k by reliability)
  * qrqfl_fixed_kyber -- HMM + DPP for clients, PQ scheme forced to Kyber

Outputs (all under ``runs/<timestamp>/``):

  * ``summary.json`` -- one entry per method with config + metrics
  * ``accuracy_curves.csv`` -- (round, method, accuracy) tidy table
  * ``per_method/<method>/metrics.csv`` -- server-side QRQ metrics if recorded
  * ``per_method/<method>/history.json`` -- raw centralized history

Run inside Colab or a local Python environment with ``flwr[simulation]``,
``torch``, ``torchvision``, ``ray``.  Use ``--smoke`` for a quick CPU sanity
run; the default config is identical to ``main.py``/the notebook so the
measured numbers can replace the legacy table directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import flwr as fl

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data.partition import dirichlet_partition  # noqa: E402
from fl.client import QRQClient                  # noqa: E402
from fl.strategy import QRQConfig, QRQFLStrategy  # noqa: E402
from hmm.reliability import HMMParams            # noqa: E402
from pqc.timing import measure as pq_measure     # noqa: E402


# -----------------------------------------------------------------------------
# Common factories
# -----------------------------------------------------------------------------

def small_cnn(num_classes: int, in_channels: int = 3) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
        nn.Linear(256, num_classes),
    )


def cifar_loaders(dataset: str, n_clients: int, alpha: float, batch: int, seed: int):
    if dataset == "cifar10":
        ds_cls, num_classes = torchvision.datasets.CIFAR10, 10
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    elif dataset == "cifar100":
        ds_cls, num_classes = torchvision.datasets.CIFAR100, 100
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    else:
        raise ValueError(dataset)
    tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    root = pathlib.Path("data_cache")
    train = ds_cls(root=root, train=True, download=True, transform=tfm)
    test = ds_cls(root=root, train=False, download=True, transform=tfm)
    labels = np.asarray(train.targets)
    splits = dirichlet_partition(labels, n_clients, alpha=alpha, seed=seed)
    train_loaders = [
        DataLoader(Subset(train, idx.tolist()), batch_size=batch, shuffle=True, num_workers=0)
        for idx in splits
    ]
    val_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=0)
    return train_loaders, val_loader, num_classes


def make_evaluate_fn(num_classes: int, val_loader, device: str):
    """Centralised evaluation - identical for every method."""
    def evaluate_fn(server_round, parameters, config):
        from flwr.common import Parameters, parameters_to_ndarrays
        if isinstance(parameters, Parameters):
            arrays = parameters_to_ndarrays(parameters)
        else:
            arrays = parameters
        model = small_cnn(num_classes).to(device)
        ckpt = {
            k: torch.tensor(arr, device=device)
            for k, arr in zip(model.state_dict().keys(), arrays)
        }
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss_sum += loss_fn(logits, yb).item()
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
        acc = correct / max(total, 1)
        return float(loss_sum / max(total, 1)), {"accuracy": float(acc)}
    return evaluate_fn


# -----------------------------------------------------------------------------
# Method registry
# -----------------------------------------------------------------------------

DEFAULT_PQ_SET = ("kyber512", "dilithium2", "falcon-512")


def method_specs() -> list[dict[str, Any]]:
    """Return the publication-comparison method grid.

    Each entry has:
      - id        : short tag used in artifacts
      - label     : human-readable label used in the paper
      - kind      : 'fedavg' (vanilla flwr FedAvg) or 'qrqfl' (our strategy)
      - pq_force  : optional fixed PQ scheme name (None means real selection
                    or no PQ overhead at all for vanilla FedAvg)
      - qrq       : kwargs forwarded into QRQConfig (only for kind='qrqfl')
    """
    return [
        # Measured baselines ---------------------------------------------------
        {"id": "fedavg",        "label": "FedAvg",
         "kind": "fedavg", "pq_force": None,
         "qrq": {}},
        {"id": "fedavg_kyber",  "label": "FedAvg + Kyber",
         "kind": "fedavg", "pq_force": "kyber512",
         "qrq": {}},
        {"id": "fedprox",       "label": "FedProx",
         "kind": "qrqfl",  "pq_force": None,
         "qrq": dict(use_hmm=False, use_dpp=False, fixed_pq=None,
                     proximal_mu=0.01)},
        # Proposed -------------------------------------------------------------
        {"id": "qrqfl",         "label": "QRQ-FL (full)",
         "kind": "qrqfl",  "pq_force": None,
         "qrq": dict(use_hmm=True, use_dpp=True)},
        # Ablations ------------------------------------------------------------
        {"id": "qrqfl_no_hmm",  "label": "QRQ-FL w/o HMM",
         "kind": "qrqfl",  "pq_force": None,
         "qrq": dict(use_hmm=False, use_dpp=True)},
        {"id": "qrqfl_no_dpp",  "label": "QRQ-FL w/o DPP",
         "kind": "qrqfl",  "pq_force": None,
         "qrq": dict(use_hmm=True, use_dpp=False)},
        {"id": "qrqfl_fixed_kyber", "label": "QRQ-FL fixed Kyber",
         "kind": "qrqfl",  "pq_force": "kyber512",
         "qrq": dict(use_hmm=True, use_dpp=True, fixed_pq="kyber512")},
    ]


# -----------------------------------------------------------------------------
# Per-method runner
# -----------------------------------------------------------------------------

def make_pq_sampler(pq_means: dict[str, float], force: str | None,
                    overhead: bool):
    """Return a sampler for the per-client PQ sleep duration.

    * vanilla FedAvg without PQ overhead -> returns 0
    * FedAvg + Kyber                     -> exponential around Kyber mean
    * QRQ-FL variants                    -> exponential around the scheme the
                                            server actually selected this round
    """
    if not overhead:
        return lambda algo: 0.0
    if force is not None:
        m = pq_means.get(force, 0.0)
        return lambda algo: float(np.random.exponential(m))
    return lambda algo: float(np.random.exponential(pq_means.get(algo, 0.0)))


def run_one_method(spec, *, n_clients, n_select, n_rounds, batch_size,
                   train_loaders, val_loader, num_classes, evaluate_fn,
                   pq_means, client_device):
    """Run a single method and return a dict of outputs."""
    overhead_on = spec["kind"] == "qrqfl" or spec["pq_force"] is not None
    sampler = make_pq_sampler(pq_means, spec["pq_force"], overhead_on)

    def client_fn(cid: str) -> fl.client.Client:
        i = int(cid)
        return QRQClient(
            client_id=cid,
            model_factory=lambda: small_cnn(num_classes),
            train_loader=train_loaders[i],
            val_loader=val_loader,
            device=client_device,
            local_epochs=1,
            pq_overhead_sampler=sampler,
        ).to_client()

    if spec["kind"] == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=float(n_select) / float(n_clients),
            fraction_evaluate=0.0,
            min_fit_clients=n_select,
            min_evaluate_clients=1,
            min_available_clients=n_clients,
            evaluate_fn=evaluate_fn,
        )
        qrq_strategy = None
    else:
        qrq_kwargs = dict(spec["qrq"])
        qrq_kwargs.setdefault("n_select", n_select)
        qrq_kwargs.setdefault("pq_set", DEFAULT_PQ_SET)
        qrq_strategy = QRQFLStrategy(
            fraction_fit=0.0, fraction_evaluate=0.0,
            min_fit_clients=n_select, min_evaluate_clients=1,
            min_available_clients=n_clients,
            evaluate_fn=evaluate_fn,
            qrq=QRQConfig(**qrq_kwargs),
            hmm=HMMParams.reasonable_default(),
        )
        strategy = qrq_strategy

    t0 = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    wall = time.time() - t0

    # Convert centralised metrics into a plain list of (round, accuracy) pairs.
    mc = getattr(history, "metrics_centralized", {}) or {}
    acc_pairs = [(int(r), float(v)) for r, v in mc.get("accuracy", [])]

    return {
        "spec": spec,
        "wall_seconds": wall,
        "accuracy_curve": acc_pairs,
        "qrq_metrics": list(getattr(qrq_strategy, "metrics", []) or []),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--select",  type=int, default=5)
    ap.add_argument("--rounds",  type=int, default=8)
    ap.add_argument("--alpha",   type=float, default=0.3)
    ap.add_argument("--batch",   type=int, default=32)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--methods", default="all",
                    help="Comma-separated method ids or 'all'")
    ap.add_argument("--smoke",   action="store_true",
                    help="Reduce to one round / few clients for a quick test")
    ap.add_argument("--out",     default=None,
                    help="Output directory (default: runs/<timestamp>)")
    args = ap.parse_args(argv)

    if args.smoke:
        args.clients = max(args.clients // 2, 4)
        args.select = max(args.select // 2, 2)
        args.rounds = 1

    # Determinism --------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    client_device = "cpu" if device == "cuda" else device

    out_dir = pathlib.Path(args.out) if args.out else (
        pathlib.Path("runs") / time.strftime("baselines_%Y%m%d_%H%M%S")
    )
    (out_dir / "per_method").mkdir(parents=True, exist_ok=True)
    print(f"Artifacts -> {out_dir.resolve()}")

    # Data, model, evaluation are shared across methods -----------------------
    train_loaders, val_loader, num_classes = cifar_loaders(
        args.dataset, args.clients, args.alpha, args.batch, args.seed
    )
    evaluate_fn = make_evaluate_fn(num_classes, val_loader, device)
    pq_means = {algo: pq_measure(algo, n=20).mean for algo in DEFAULT_PQ_SET}
    print("PQ mean service times (s):", json.dumps(pq_means, indent=2))

    selected_ids = (
        [s["id"] for s in method_specs()] if args.methods == "all"
        else [s.strip() for s in args.methods.split(",") if s.strip()]
    )
    methods = [s for s in method_specs() if s["id"] in selected_ids]
    if not methods:
        raise SystemExit(f"No methods matched: {args.methods}")

    summary: dict[str, Any] = {
        "config": {
            "dataset": args.dataset,
            "n_clients": args.clients,
            "n_select": args.select,
            "n_rounds": args.rounds,
            "batch_size": args.batch,
            "alpha": args.alpha,
            "seed": args.seed,
            "device": device,
            "pq_set": list(DEFAULT_PQ_SET),
            "pq_means_seconds": pq_means,
        },
        "methods": [],
    }

    accuracy_rows: list[dict[str, Any]] = []

    for spec in methods:
        print(f"\n=== Running method: {spec['id']} ({spec['label']}) ===")
        result = run_one_method(
            spec,
            n_clients=args.clients,
            n_select=args.select,
            n_rounds=args.rounds,
            batch_size=args.batch,
            train_loaders=train_loaders,
            val_loader=val_loader,
            num_classes=num_classes,
            evaluate_fn=evaluate_fn,
            pq_means=pq_means,
            client_device=client_device,
        )

        method_dir = out_dir / "per_method" / spec["id"]
        method_dir.mkdir(parents=True, exist_ok=True)
        with (method_dir / "history.json").open("w") as f:
            json.dump({
                "label": spec["label"],
                "accuracy_curve": result["accuracy_curve"],
                "wall_seconds": result["wall_seconds"],
            }, f, indent=2)
        if result["qrq_metrics"]:
            with (method_dir / "metrics.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["round", "n_received", "n_failed", "rho_mean",
                            "Q0", "Q1", "Q2", "pq_scheme", "ts"])
                for m in result["qrq_metrics"]:
                    w.writerow([m["round"], m["n_received"], m["n_failed"],
                                m["rho_mean"], *m["Q"], m["pq_scheme"], m["ts"]])

        for r, a in result["accuracy_curve"]:
            accuracy_rows.append({"method_id": spec["id"], "label": spec["label"],
                                  "round": r, "accuracy": a})

        final_acc = result["accuracy_curve"][-1][1] if result["accuracy_curve"] else None
        summary["methods"].append({
            "id": spec["id"],
            "label": spec["label"],
            "kind": spec["kind"],
            "qrq": spec["qrq"],
            "pq_force": spec["pq_force"],
            "wall_seconds": result["wall_seconds"],
            "final_accuracy": final_acc,
            "rounds_run": len(result["accuracy_curve"]),
        })

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "accuracy_curves.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method_id", "label", "round", "accuracy"])
        w.writeheader()
        w.writerows(accuracy_rows)

    print("\n--- Summary ---")
    for m in summary["methods"]:
        acc = "n/a" if m["final_accuracy"] is None else f"{m['final_accuracy']:.4f}"
        print(f"  {m['id']:<22} final_acc={acc}  wall={m['wall_seconds']:.1f}s")
    print(f"\nWrote {out_dir / 'summary.json'} and {out_dir / 'accuracy_curves.csv'}")


if __name__ == "__main__":
    main()
