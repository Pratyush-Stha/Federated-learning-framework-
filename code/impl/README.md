# QRQ-FL implementation (`code/impl`)

This directory is the **Python package and experiments** for QRQ-FL.

**Start here:** `QRQFL_colab.ipynb` — same stack as `main.py`, runs on **Google Colab** or locally.

## Module map

```
code/impl/
├── QRQFL_colab.ipynb      # main entry (Colab + local)
├── main.py                # Flower CLI (advanced)
├── simulate_results.py    # fast analytical-only simulator
├── requirements.txt
├── _build_notebook.py     # regenerates the notebook from sources below
├── _validate_notebook.py  # syntax-check notebook code cells
├── hmm/
│   └── reliability.py     # 4-state HMM, forward, Baum–Welch, dropout_probability
├── queueing/
│   └── multistage_queue.py
├── pqc/
│   └── timing.py          # liboqs or stub cycle model
├── rl/
│   └── dpp.py             # DPP selector + virtual queues
├── data/
│   ├── partition.py       # Dirichlet, FEMNIST helpers, etc.
│   └── iov_telemetry.py
└── fl/
    ├── strategy.py        # QRQFLStrategy (FedAvg + hooks)
    └── client.py          # QRQClient (NumPyClient)
```

## How to run

### 1. Colab (recommended)

From the **repository root** on GitHub: Colab **File → Open notebook → GitHub** → open **`code/impl/QRQFL_colab.ipynb`**.

Enable a **GPU** runtime if available. Run the **first cell** first (installs **`flwr[simulation]`** and **Ray**). If you change installs, use **Runtime → Restart runtime**, then **Run all**.

### 2. Local notebook

```bash
cd code/impl   # from repo root

python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Unix:    source .venv/bin/activate

pip install -r requirements.txt jupyter ipykernel
# Optional CPU torch:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

jupyter notebook QRQFL_colab.ipynb
```

### 3. CLI (`main.py`)

```bash
cd code/impl
pip install -r requirements.txt
python main.py --dataset cifar100 --clients 50 --rounds 30 --alpha 0.3 --batch 64 --schedule dpp
```

Flower + Ray can be flaky on some Windows setups; prefer Colab if you hit worker issues.

### 4. Analytical-only (`simulate_results.py`)

```bash
cd code/impl
python simulate_results.py
```

No full FL training—useful for quick queue/HMM sanity checks.

## Datasets

| Dataset | Auto-download | Notes |
|---------|---------------|--------|
| CIFAR-10 | Yes (`torchvision`) | Quick experiments |
| CIFAR-100 | Yes (`torchvision`) | Paper-aligned default in notebook |
| FEMNIST | No | LEAF preprocessing required |
| MIMIC-III | No | Credentials / data not bundled |

## Smoke tests

```bash
cd code/impl
python hmm/reliability.py
python queueing/multistage_queue.py
python data/partition.py
python pqc/timing.py
```

## Limitations

- **`fl.simulation.start_simulation` needs Ray** — install **`flwr[simulation]`** (see `requirements.txt`).
- **`liboqs-python`** is optional; without it, `pqc/timing.py` uses deterministic stub timings.
- Secure aggregation in the paper sense is **not** fully implemented as real crypto; clients simulate PQ delay for scheduling experiments.

## Rebuild the notebook from sources

After editing `.py` files:

```bash
cd code/impl
python _build_notebook.py
python _validate_notebook.py
```
