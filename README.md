# QRQ-FL: Quantum-Resilient, Queue-Aware, Reliability-Adaptive Federated Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPONAME/blob/main/impl/QRQFL_colab.ipynb)

> Replace `USERNAME/REPONAME` in the badge above with your actual GitHub
> path after the first push. That is the only edit required before the
> "Open in Colab" button works.

This repository hosts the **runnable code** for the QRQ-FL federated-learning
framework. It unifies three layers under one Drift-Plus-Penalty controller:

1. **Hidden Markov client reliability** (Online / Busy / Offline / Byzantine)
2. **Multi-stage queueing** (M/G/1 wireless uplink and PQ encryption,
   M/M/c edge aggregation, M/G/1 blockchain validation)
3. **Post-quantum cryptographic agility** (Kyber, Dilithium, Falcon,
   FrodoKEM, NTRU)

The paper itself (`main.tex`) and internal reports are kept **local only**
(see `.gitignore`). What you see on GitHub is exactly what Colab needs to
reproduce the results.

## Quick start (Google Colab)

1. **Push this repo to GitHub** (instructions below)
2. Edit the badge URL above to point at your repo
3. Click the badge — the notebook opens in Colab
4. **Runtime ▸ Change runtime type** and pick a GPU:
   - **A100 (Colab Pro+)** — fastest, ~5 min for CIFAR-100 full run
   - **V100 (Colab Pro)** — ~8 min
   - **T4 (free / Pro)** — ~10–15 min for CIFAR-10, ~30 min for CIFAR-100
   - CPU — works but slow; only use for code review
5. **Runtime ▸ Run all**
6. When the last cell finishes, `figures.zip` and `results.json` are
   downloaded to your machine automatically

The notebook auto-detects Colab vs. local, installs the small extras it
needs (`hmmlearn`, `simpy`, `tqdm`), and uses whatever device is available.

## Quick start (local CPU/GPU)

```bash
git clone https://github.com/USERNAME/REPONAME.git
cd REPONAME/impl

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt jupyter ipykernel

# CPU-only torch (small, fast install)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

jupyter notebook QRQFL_colab.ipynb
```

The same notebook works on Colab and local installs.

## Repository layout

```
.
├── README.md                      # this file
├── LICENSE                        # MIT
├── .gitignore                     # excludes paper sources, datasets, caches
└── impl/
    ├── QRQFL_colab.ipynb          # ⭐ main entry point (Colab + local)
    ├── _build_notebook.py         # rebuilds the notebook from Python source
    ├── _validate_notebook.py      # AST-checks every code cell
    ├── README.md
    ├── requirements.txt
    ├── main.py                    # advanced Flower-based CLI driver
    ├── simulate_results.py        # pure analytical fast simulator (no torch)
    ├── hmm/reliability.py         # 4-state HMM + forward + Baum-Welch
    ├── queueing/multistage_queue.py  # Pollaczek-Khinchine + Erlang-C + SimPy
    ├── pqc/timing.py              # liboqs benchmark (or stub fallback)
    ├── rl/dpp.py                  # Drift-Plus-Penalty selector
    ├── data/partition.py          # Dirichlet, FEMNIST, hospital partitioners
    ├── data/iov_telemetry.py      # synthetic IoV generator
    └── fl/                        # Flower strategy + client (optional)
```

## What runs out of the box

| Component | Auto-runs in Colab? | Notes |
|---|---|---|
| CIFAR-10 / CIFAR-100 training | Yes (torchvision) | 170 MB auto-download |
| HMM reliability + Baum-Welch | Yes | Pure NumPy |
| Queue analytics (M/G/1, M/M/c) | Yes | Pure NumPy |
| Queue simulation (SimPy) | Yes | Auto-installed in cell 2 |
| DPP scheduler | Yes | Pure NumPy |
| FEMNIST | No | Run [LEAF](https://github.com/TalwalkarLab/leaf) preprocessing locally |
| MIMIC-III | No | PhysioNet credentials required |
| **Real PQ encryption (`liboqs`)** | No | Difficult to build on Colab; falls back to a deterministic cycle-count stub |
| Flower simulation backend | Optional | Not required by the notebook |

## First push to GitHub

```bash
cd papers
git init
git add .
git status                # confirm main.tex, *.pdf, *.docx are NOT listed
git commit -m "QRQ-FL implementation and Colab notebook"
git branch -M main
git remote add origin https://github.com/USERNAME/REPONAME.git
git push -u origin main
```

After the push, edit the badge URL in this file (`USERNAME/REPONAME`) and
push again. The Colab badge then works for anyone who visits the repo.

## License

MIT — see `LICENSE`.
