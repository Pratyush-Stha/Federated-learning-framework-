"""Build QRQFL_colab.ipynb: full Flower + HMM + DPP + PQ stack, real CIFAR.

Run:  python _build_notebook.py
"""
from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "QRQFL_colab.ipynb"


def strip_main_block(src: str) -> str:
    """Remove trailing if __name__ == '__main__': block so notebook cells don't run demos."""
    m = re.search(r"\nif __name__ == [\"']__main__[\"']:\n", src)
    if m:
        return src[: m.start()].rstrip() + "\n"
    return src


def read_module(rel: str) -> str:
    return strip_main_block((ROOT / rel).read_text(encoding="utf-8"))


def strip_strategy_imports(src: str) -> str:
    out = []
    for line in src.splitlines():
        if line.startswith("from hmm.") or line.startswith("from rl.") or line.startswith("from pqc."):
            continue
        out.append(line)
    return "\n".join(out)


cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text,
        }
    )


# ── 0: title ──────────────────────────────────────────────────────────
md(
    """# QRQ-FL — full implementation (Colab)

This notebook runs the **same stack as `main.py`** end-to-end:

* **Real vision data**: CIFAR-10 or CIFAR-100 (torchvision download).
* **Non-IID split**: Dirichlet label skew (`data/partition.py`).
* **Flower simulation**: `fl.simulation.start_simulation`.
* **Clients**: `QRQClient` — local SGD + simulated PQ delay from measured/stub timings.
* **Server**: `QRQFLStrategy` — HMM dropout beliefs (`hmm/reliability.py`), DPP client + PQ selection (`rl/dpp.py`), virtual queues, FedAvg aggregation.

You get a **FedAvg** baseline (random client sampling) and **QRQ-FL** (DPP + HMM + PQ set), with **centralized test accuracy each round**, metric CSV, plots, and `results.json`.

**Colab:** enable GPU (Runtime → Change runtime type). The last cell downloads `figures.zip` and `results.json` when `google.colab` is available.

**Note:** `fl.simulation.start_simulation` needs **Ray**. This notebook installs `flwr[simulation]` (not plain `flwr`) so Ray is pulled in automatically.
"""
)

# ── 1: install ────────────────────────────────────────────────────────
md("## 1. Environment and dependencies")
code(
    """import sys, subprocess

COLAB = "google.colab" in sys.modules
print(f"Running on: {'Google Colab' if COLAB else 'local'}")
print(f"Python:     {sys.version.split()[0]}")


def pip_install(pkg: str) -> None:
    print(f"  installing {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def have_ray() -> bool:
    try:
        import ray  # noqa: F401
        return True
    except ImportError:
        return False


# Flower simulation backend requires Ray — use the official extra (see flwr error message).
FLWR_SIM = "flwr[simulation]>=1.8.0,<2"
try:
    import flwr  # noqa: F401
    if not have_ray():
        pip_install(FLWR_SIM)
except ImportError:
    pip_install(FLWR_SIM)

for pkg in ("matplotlib", "pandas", "tqdm"):
    try:
        if pkg == "matplotlib":
            import matplotlib  # noqa: F401
        elif pkg == "pandas":
            import pandas  # noqa: F401
        else:
            import tqdm  # noqa: F401
    except ImportError:
        pip_install(pkg)

import torch
print(f"torch:      {torch.__version__}")
import flwr as fl
print(f"flwr:       {fl.__version__}")
if not have_ray():
    raise RuntimeError("Ray is still missing after installing flwr[simulation]. Run: pip install -U 'flwr[simulation]'")
import ray
print(f"ray:        {ray.__version__}")
"""
)

# ── 2: imports + device ───────────────────────────────────────────────
md("## 2. Imports, seed, device")
code(
    """from __future__ import annotations

import csv
import json
import pathlib
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_TIER = "cpu"
if DEVICE == "cuda":
    name = torch.cuda.get_device_name(0).lower()
    if "a100" in name:
        GPU_TIER = "a100"
    elif "v100" in name:
        GPU_TIER = "v100"
    elif "l4" in name:
        GPU_TIER = "l4"
    elif "t4" in name:
        GPU_TIER = "t4"
    else:
        GPU_TIER = "gpu_other"
    print(f"Device: {DEVICE} | {torch.cuda.get_device_name(0)} | tier={GPU_TIER}")
else:
    print(f"Device: {DEVICE} (no GPU)")
"""
)

# ── 3: hyperparameters ────────────────────────────────────────────────
md(
    """## 3. Hyperparameters

Federation size defaults scale with `GPU_TIER`. Override by editing the dict literals.
"""
)
code(
    """# --- Dataset ---
DATASET = "cifar100"  # "cifar10" | "cifar100"

PRESETS = {
    "cpu":       dict(N_CLIENTS=10, N_SELECT=5,  N_ROUNDS=8,  BATCH_SIZE=32),
    "t4":        dict(N_CLIENTS=20, N_SELECT=10, N_ROUNDS=15, BATCH_SIZE=64),
    "l4":        dict(N_CLIENTS=30, N_SELECT=12, N_ROUNDS=20, BATCH_SIZE=128),
    "v100":      dict(N_CLIENTS=40, N_SELECT=12, N_ROUNDS=25, BATCH_SIZE=128),
    "a100":      dict(N_CLIENTS=50, N_SELECT=15, N_ROUNDS=30, BATCH_SIZE=256),
    "gpu_other": dict(N_CLIENTS=20, N_SELECT=10, N_ROUNDS=15, BATCH_SIZE=64),
}
preset = PRESETS[GPU_TIER]
N_CLIENTS = preset["N_CLIENTS"]
N_SELECT = preset["N_SELECT"]
N_ROUNDS = preset["N_ROUNDS"]
BATCH_SIZE = preset["BATCH_SIZE"]

DIRICHLET_ALPHA = 0.3
LOCAL_EPOCHS = 1
LR = 0.01
MOMENTUM = 0.9

# DPP / QRQ (matches main.py defaults)
DPP_V = 2.0
PQ_SET = ("kyber512", "dilithium2", "falcon-512")
QRQ_DEADLINE_S = 2.0

# Run FedAvg baseline first (doubles wall time; set False for QRQ-FL only)
RUN_FEDAVG_BASELINE = True

print(json.dumps({
    "DATASET": DATASET,
    "N_CLIENTS": N_CLIENTS,
    "N_SELECT": N_SELECT,
    "N_ROUNDS": N_ROUNDS,
    "BATCH_SIZE": BATCH_SIZE,
    "DIRICHLET_ALPHA": DIRICHLET_ALPHA,
    "DEVICE": DEVICE,
}, indent=2))
"""
)

# ── 4: partition + dataset ───────────────────────────────────────────
md("## 4. Real dataset + Dirichlet non-IID split (`data/partition.py`)")
code(
    read_module("data/partition.py")
    + """

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

if DATASET == "cifar10":
    NUM_CLASSES = 10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    DS = torchvision.datasets.CIFAR10
elif DATASET == "cifar100":
    NUM_CLASSES = 100
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    DS = torchvision.datasets.CIFAR100
else:
    raise ValueError(DATASET)

tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
print(f"Downloading {DATASET} …")
train = DS(root=DATA_DIR, train=True, download=True, transform=tfm)
test = DS(root=DATA_DIR, train=False, download=True, transform=tfm)
labels = np.asarray(train.targets)
splits = dirichlet_partition(labels, N_CLIENTS, alpha=DIRICHLET_ALPHA, seed=SEED)
train_loaders = [
    DataLoader(Subset(train, idx.tolist()), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    for idx in splits
]
val_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=0)
sizes = [len(s) for s in splits]
print(f"Train {len(train):,} | Test {len(test):,} | clients min/med/max = {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")
"""
)

# ── 5: model ──────────────────────────────────────────────────────────
md("## 5. CNN (same architecture as `main.py`)")
code(
    """def small_cnn(num_classes: int, in_channels: int = 3) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
        nn.Linear(256, num_classes),
    )


IN_CHANNELS = 3
model_example = small_cnn(NUM_CLASSES, IN_CHANNELS).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model_example.parameters()):,}")
del model_example
"""
)

# ── 6–10: library modules ─────────────────────────────────────────────
md("## 6. HMM reliability layer (`hmm/reliability.py`)")
code("# --- inlined from hmm/reliability.py ---\n" + read_module("hmm/reliability.py"))

md("## 7. PQ timing (`pqc/timing.py`)")
code("# --- inlined from pqc/timing.py ---\n" + read_module("pqc/timing.py"))

md("## 8. DPP scheduler (`rl/dpp.py`)")
code("# --- inlined from rl/dpp.py ---\n" + read_module("rl/dpp.py"))

md("## 9. Flower client (`fl/client.py`)")
code("# --- inlined from fl/client.py ---\n" + read_module("fl/client.py"))

md("## 10. QRQ-FL strategy (`fl/strategy.py`)")
code(
    "# --- inlined from fl/strategy.py (package imports omitted; see cells above) ---\n"
    "measure_pq = measure  # alias used inside QRQFLStrategy\n"
    + strip_strategy_imports(read_module("fl/strategy.py"))
)

# ── 11: PQ profile, clients, eval, run ───────────────────────────────
md("## 11. PQ profile, client factory, centralized evaluation")
code(
    """# Mean PQ service times (liboqs if available, else deterministic stub)
pq_means: dict[str, float] = {algo: measure(algo, n=20).mean for algo in PQ_SET}
print("PQ mean service times (s):", json.dumps(pq_means, indent=2))


def pq_sleep_sample(algo: str) -> float:
    return float(np.random.exponential(pq_means.get(algo, 0.01)))


def client_fn(cid: str) -> fl.client.Client:
    i = int(cid)
    return QRQClient(
        client_id=cid,
        model_factory=lambda: small_cnn(NUM_CLASSES, IN_CHANNELS),
        train_loader=train_loaders[i],
        val_loader=val_loader,
        device=DEVICE,
        local_epochs=LOCAL_EPOCHS,
        pq_overhead_sampler=pq_sleep_sample,
    ).to_client()


def make_evaluate_fn():
    def evaluate_fn(server_round: int, parameters, config: dict):
        if isinstance(parameters, Parameters):
            arrays = parameters_to_ndarrays(parameters)
        else:
            arrays = parameters  # some flwr versions pass ndarrays directly
        model = small_cnn(NUM_CLASSES, IN_CHANNELS).to(DEVICE)
        ckpt = {
            k: torch.tensor(arr, device=DEVICE)
            for k, arr in zip(model.state_dict().keys(), arrays)
        }
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss_sum += loss_fn(logits, yb).item()
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
        acc = correct / max(total, 1)
        loss = loss_sum / max(total, 1)
        return float(loss), {"accuracy": float(acc)}

    return evaluate_fn


evaluate_fn = make_evaluate_fn()
"""
)

md("## 12. Run simulations")
code(
    """RUN_DIR = pathlib.Path("runs") / time.strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"Artifacts -> {RUN_DIR.resolve()}")

hist_fedavg = None
if RUN_FEDAVG_BASELINE:
    print("\\n=== FedAvg baseline (random client sampling) ===")
    strat_fed = fl.server.strategy.FedAvg(
        fraction_fit=float(N_SELECT) / float(N_CLIENTS),
        fraction_evaluate=0.0,
        min_fit_clients=N_SELECT,
        min_evaluate_clients=1,
        min_available_clients=N_CLIENTS,
        evaluate_fn=evaluate_fn,
    )

    def client_fn_plain(cid: str) -> fl.client.Client:
        i = int(cid)
        return QRQClient(
            client_id=cid,
            model_factory=lambda: small_cnn(NUM_CLASSES, IN_CHANNELS),
            train_loader=train_loaders[i],
            val_loader=val_loader,
            device=DEVICE,
            local_epochs=LOCAL_EPOCHS,
            pq_overhead_sampler=lambda _a: 0.0,
        ).to_client()

    hist_fedavg = fl.simulation.start_simulation(
        client_fn=client_fn_plain,
        num_clients=N_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strat_fed,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    print(hist_fedavg)

print("\\n=== QRQ-FL (HMM + DPP + PQ library) ===")
strategy_qrq = QRQFLStrategy(
    fraction_fit=0.0,
    fraction_evaluate=0.0,
    min_fit_clients=N_SELECT,
    min_evaluate_clients=1,
    min_available_clients=N_CLIENTS,
    evaluate_fn=evaluate_fn,
    qrq=QRQConfig(n_select=N_SELECT, V=DPP_V, pq_set=tuple(PQ_SET), deadline_s=QRQ_DEADLINE_S),
    hmm=HMMParams.reasonable_default(),
)

hist_qrq = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=N_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
    strategy=strategy_qrq,
    client_resources={"num_cpus": 1, "num_gpus": 0.0},
)
print(hist_qrq)

# Save server-side QRQ metrics (same shape as main.py CSV)
with (RUN_DIR / "metrics_qrq.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round", "n_received", "n_failed", "rho_mean", "Q0", "Q1", "Q2", "pq_scheme", "ts"])
    for m in strategy_qrq.metrics:
        w.writerow(
            [
                m["round"],
                m["n_received"],
                m["n_failed"],
                m["rho_mean"],
                *m["Q"],
                m["pq_scheme"],
                m["ts"],
            ]
        )
print(f"Wrote {RUN_DIR / 'metrics_qrq.csv'}")
"""
)

# ── 13: extract metrics + plots ───────────────────────────────────────
md("## 13. Accuracy curves and QRQ metrics")
code(
    """def series_accuracy(hist) -> tuple[list[int], list[float]]:
    if hist is None:
        return [], []
    mc = getattr(hist, "metrics_centralized", None) or {}
    acc = mc.get("accuracy", [])
    rounds = [int(r) for r, _ in acc]
    vals = [float(v) for _, v in acc]
    return rounds, vals


r2, a2 = series_accuracy(hist_qrq)
plt.figure(figsize=(7, 4))
plt.plot(r2, a2, marker="s", markersize=4, label="QRQ-FL")
if hist_fedavg is not None:
    r1, a1 = series_accuracy(hist_fedavg)
    plt.plot(r1, a1, marker="o", markersize=4, label="FedAvg")
plt.xlabel("Round")
plt.ylabel("Test accuracy")
plt.title(f"{DATASET.upper()} | N={N_CLIENTS}, K={N_SELECT}, α={DIRICHLET_ALPHA}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
FIG_DIR = pathlib.Path("figures")
FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / "acc_centralized.pdf")
plt.savefig(IMG_PNG := FIG_DIR / "acc_centralized.png", dpi=150)
plt.show()

# PQ scheme picked per round
if strategy_qrq.metrics:
    rounds = [m["round"] for m in strategy_qrq.metrics]
    schemes = [m["pq_scheme"] for m in strategy_qrq.metrics]
    rho = [m["rho_mean"] for m in strategy_qrq.metrics]
    dfm = pd.DataFrame({"round": rounds, "pq_scheme": schemes, "rho_mean": rho})
    print(dfm.tail(10).to_string())
"""
)

# ── 14: results.json + Colab download ────────────────────────────────
md("## 14. `results.json` and downloads")
code(
    """def acc_dict(hist):
    _, vals = series_accuracy(hist)
    return vals


summary = {
    "config": {
        "dataset": DATASET,
        "n_clients": N_CLIENTS,
        "n_select": N_SELECT,
        "n_rounds": N_ROUNDS,
        "batch_size": BATCH_SIZE,
        "alpha": DIRICHLET_ALPHA,
        "local_epochs": LOCAL_EPOCHS,
        "lr": LR,
        "dpp_V": DPP_V,
        "pq_set": list(PQ_SET),
        "device": DEVICE,
        "seed": SEED,
    },
    "fedavg_test_acc_per_round": acc_dict(hist_fedavg) if hist_fedavg else [],
    "qrqfl_test_acc_per_round": acc_dict(hist_qrq),
    "qrqfl_server_metrics": strategy_qrq.metrics,
    "pq_mean_seconds": pq_means,
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("Wrote results.json")

if COLAB:
    import shutil

    shutil.make_archive("figures", "zip", str(FIG_DIR))
    shutil.make_archive("run_artifacts", "zip", str(RUN_DIR))
    from google.colab import files

    files.download("figures.zip")
    files.download("run_artifacts.zip")
    files.download("results.json")
else:
    print("Local run: see figures/, results.json, and runs/")
"""
)

# ── write notebook ───────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0",
        },
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
