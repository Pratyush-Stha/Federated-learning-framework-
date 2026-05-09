# Running QRQ-FL (Local + Google Colab)

This repo’s **primary entry point** is the notebook:

- `code/impl/QRQFL_colab.ipynb`

It runs on **Google Colab** and **locally**. The notebook downloads a **real dataset** (CIFAR-10/100) via `torchvision`, partitions it non‑IID, and runs the full QRQ‑FL stack.

---

## Local run (Windows 10/11, PowerShell)

### Prerequisites

- **Python 3.10+** installed (recommended: 3.10 or 3.11)
- `git` installed (optional but recommended)
- (Optional) **NVIDIA GPU + CUDA drivers** if you want GPU locally  
  If you don’t have GPU, CPU works (just slower).

### 1) Get the code

If you already have this folder, you can skip cloning.

```powershell
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2) Create and activate a virtual environment

```powershell
cd code\impl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked (common on Windows), run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then re-run:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3) Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter ipykernel
```

### 4) Install PyTorch + torchvision

#### Option A (recommended for “it just works”): CPU-only wheels

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Option B: GPU wheels

Install the CUDA-enabled wheels that match your CUDA runtime. The most reliable way is to follow the official selector:

- PyTorch install page: `https://pytorch.org/get-started/locally/`

After installing, you can verify GPU availability:

```powershell
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available())"
```

### 5) Launch the notebook

```powershell
jupyter notebook QRQFL_colab.ipynb
```

In the browser UI, open `QRQFL_colab.ipynb` and click **Kernel → Restart & Run All**.

### 6) Outputs you should see

After a full run you should have:

- `code/impl/figures/` (plots, e.g. `acc_centralized.png/pdf`)
- `code/impl/results.json`
- `code/impl/runs/<timestamp>/metrics_qrq.csv` (server-side QRQ metrics)

### 7) (Optional) Rebuild the notebook from the source modules

If you edit any module under `code/impl/` and want the notebook to be regenerated consistently:

```powershell
python _build_notebook.py
python _validate_notebook.py
```

### Common local issues (Windows)

- **`Activate.ps1` cannot be loaded**: use `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` as shown above.
- **Very slow run**: reduce federation scale in the notebook hyperparameters (e.g. fewer clients/rounds) or use Colab GPU.
- **Flower simulation/Ray quirks on Windows**: the notebook uses Flower simulation; if you hit hanging/worker issues locally, Colab is usually the quickest path.

---

## Google Colab run (every step)

You have 3 common ways to open the notebook in Colab. Pick one.

### Option A (best): Open from GitHub

1. Push this repo to GitHub (or any public Git host Colab can access).
2. In Colab: **File → Open notebook → GitHub**.
3. Paste your repo URL and open `code/impl/QRQFL_colab.ipynb`.
   (After the repo re-org, the path is `code/impl/QRQFL_colab.ipynb`.)

### Option B: Upload just the notebook

1. Go to Colab: `https://colab.research.google.com/`
2. **File → Upload notebook**
3. Upload `code/impl/QRQFL_colab.ipynb`
   (After the repo re-org, the file is at `code/impl/QRQFL_colab.ipynb`.)

Note: if you upload only the notebook (without the repo files), it will still run because the notebook **inlines** the needed modules, but you won’t have the repo structure for saving extra artifacts the same way.

### Option C: Upload the whole folder to Colab runtime

1. Open a new Colab notebook (blank is fine).
2. In the left sidebar **Files**, click **Upload**.
3. Upload the whole `code/impl/` folder contents (or a zip you extract).
4. Open and run `QRQFL_colab.ipynb`.

### 1) Enable GPU

In Colab:

- **Runtime → Change runtime type → Hardware accelerator → GPU → Save**

### 2) Run all cells

- **Runtime → Run all**

The notebook will:

- install missing pip packages (Flower + plotting stack)
- download CIFAR-10/100 via `torchvision`
- run a baseline (optional) and QRQ‑FL
- write plots and `results.json`
- if Colab is detected, download zip artifacts at the end

### 3) Expected downloads at the end

When the last cell finishes, Colab should download:

- `figures.zip`
- `run_artifacts.zip`
- `results.json`

If downloads don’t trigger, you can manually download from the Colab **Files** pane.

### Common Colab issues

- **Out of memory / too slow**: lower `N_CLIENTS`, `N_ROUNDS`, or batch size in the notebook hyperparameters cell.
- **GPU not used**: verify the GPU runtime is enabled, then re-run the “device” cell. It should print `Device: cuda`.

---

## What to run (recommended)

- **Most users**: run `code/impl/QRQFL_colab.ipynb` on Colab GPU.
- After the repo re-org: run `code/impl/QRQFL_colab.ipynb`.
- **Local**: run the same notebook using the “Local run” steps above.

---

## Advanced: command-line Flower driver (optional)

There is also a CLI entry point:

- `code/impl/main.py`
- After the repo re-org: `code/impl/main.py`

Example (from `code/impl/` with an active venv):
After the repo re-org: run it from `code/impl/`.

```powershell
pip install "flwr[simulation]>=1.8.0,<2"
python main.py --dataset cifar100 --clients 50 --rounds 20 --alpha 0.3 --batch 64 --schedule dpp
```

If you only want “the full implementation with dataset and everything”, the notebook is the intended path.

