# QRQ-FL: Quantum-Resilient, Queue-Aware, Reliability-Adaptive Federated Learning

Runnable **federated-learning** code: HMM-based reliability beliefs, DPP-style scheduling, post-quantum overhead modeling, and Flower simulation on **CIFAR-10 / CIFAR-100**.

## Open in Google Colab (no README edits required)

After this repository is on GitHub:

1. Open [Google Colab](https://colab.research.google.com/).
2. **File → Open notebook → GitHub**.
3. Sign in / authorize if prompted, then select **this repository**.
4. Open **`code/impl/QRQFL_colab.ipynb`**.
5. **Runtime → Change runtime type → GPU** (recommended).
6. **Runtime → Run all**.

The first code cell installs dependencies, including **`flwr[simulation]`** (pulls in **Ray**). `fl.simulation.start_simulation` will fail if Ray is missing—always run that cell first; use **Runtime → Restart runtime** after installs if Colab suggests it.

Optional: upload the same `.ipynb` via **File → Upload notebook** (the notebook inlines the modules; you do not need the rest of the tree on disk for a full run).

## Quick start (local)

From the repository root:

```bash
cd code/impl

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt jupyter ipykernel

# CPU-only PyTorch (optional; adjust for CUDA if you use a GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

jupyter notebook QRQFL_colab.ipynb
```

`requirements.txt` includes **`flwr[simulation]`** so **Ray** is available for Flower simulation.

## Repository layout

```
.
├── README.md                 # project overview (shown on GitHub home)
├── LICENSE
├── .gitignore
├── code/
│   └── impl/                 # all Python code, notebook, requirements
│       ├── QRQFL_colab.ipynb # main entry (Colab + local)
│       ├── requirements.txt
│       ├── main.py
│       ├── _build_notebook.py
│       ├── _validate_notebook.py
│       ├── hmm/  pqc/  rl/  fl/  data/  queueing/
│       └── README.md         # module-level notes (shown on GitHub in this folder)
├── related works/            # optional local reference PDFs (often gitignored)
└── current works in progress/  # drafts / notes (many patterns gitignored; see .gitignore)
```

Details for each package live in **`code/impl/README.md`**.

## What runs in the notebook

| Piece | Notes |
|------|--------|
| CIFAR-10 / CIFAR-100 | Downloaded via `torchvision` |
| Non-IID split | Dirichlet partitioning (`code/impl/data/partition.py`) |
| Flower simulation | `fl.simulation.start_simulation` (**requires Ray** via `flwr[simulation]`) |
| HMM / DPP / PQ timing | Inlined from `hmm/`, `rl/`, `pqc/` (rebuild with `_build_notebook.py`) |
| `liboqs` | Optional; stub timings used if not installed |

## License

MIT — see `LICENSE`.
