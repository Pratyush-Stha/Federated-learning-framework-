# `impl/` � QRQ-FL Implementation

This folder contains the runnable Python code for the QRQ-FL paper.
The cleanest entry point is **`QRQFL_colab.ipynb`** � it runs identically
on Google Colab (free T4 GPU) and on a local CPU.

## Module map

```
impl/
??? QRQFL_colab.ipynb           # ? start here
??? main.py                     # Flower CLI driver (advanced; needs flwr)
??? simulate_results.py         # analytical-only fast simulator (no torch)
??? requirements.txt
?
??? hmm/
?   ??? reliability.py          # 4-state HMM, forward algorithm,
?                                  Baum-Welch EM, dropout_probability()
?
??? queueing/
?   ??? multistage_queue.py     # Pollaczek-Khinchine M/G/1, Erlang-C,
?                                  M/M/c, SimPy event simulation
?
??? pqc/
?   ??? timing.py               # liboqs benchmark or deterministic stub
?
??? rl/
?   ??? dpp.py                  # Drift-Plus-Penalty action selector
?
??? data/
?   ??? partition.py            # Dirichlet, FEMNIST per-writer,
?   ?                              MIMIC-III hospital partitioners
?   ??? iov_telemetry.py        # synthetic IoV telemetry generator
?
??? fl/
    ??? strategy.py             # Flower QRQFLStrategy
    ??? client.py               # Flower QRQClient
```

## Three ways to run

### 1. Colab notebook (recommended)
Click **Open in Colab** in the top-level README, set runtime to T4 GPU,
Run All. The notebook installs `hmmlearn` + `simpy` automatically and
downloads CIFAR-100 / CIFAR-10 via torchvision.

### 2. Local notebook
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows
pip install -r requirements.txt jupyter ipykernel
jupyter notebook QRQFL_colab.ipynb
```

### 3. Command-line driver (Flower-based)
```bash
pip install -r requirements.txt flwr
python main.py --dataset cifar100 --clients 50 --rounds 30 \
               --pq kyber512 --schedule dpp
```
Flower uses Ray which has occasional Windows quirks. If it hangs, prefer
the notebook.

### 4. Pure analytical simulation (no PyTorch)
```bash
python simulate_results.py
```
Runs in seconds. Generates analytic curves only � does not train any
model. Useful for sanity-checking the queue and HMM math.

## Datasets

| Dataset | Auto-download | Notes |
|---|---|---|
| CIFAR-10 | ? via torchvision | recommended for quick runs |
| CIFAR-100 | ? via torchvision | matches paper text |
| MNIST | ? via torchvision | for pipeline sanity-checks |
| FEMNIST | ? | Run `cd leaf/data/femnist && ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample` from [LEAF](https://github.com/TalwalkarLab/leaf) |
| MIMIC-III | ? | Requires PhysioNet credentials; not bundled |
| IoV telemetry | ? generated locally by `data/iov_telemetry.py` | synthetic |

## Smoke-tests for individual modules

```bash
python hmm/reliability.py        # prints recovered HMM transition diag
python queueing/multistage_queue.py   # prints PK / Erlang-C numbers
python data/partition.py         # prints Dirichlet split sizes
python pqc/timing.py             # liboqs measurements (or stub fallback)
```

## Known limitations

- The Flower / Ray simulation backend is sometimes flaky on Windows.
  Prefer the notebook or use Colab.
- `liboqs-python` is hard to build on Windows; if it fails, `pqc/timing.py`
  silently uses a deterministic cycle-count stub. The numbers it returns
  are derived from public OQS benchmarks, not freshly measured.
- The `fl/` package needs `flwr`; not installed by default.

## What is real vs. analytical

The notebook clearly separates the two. Concretely:

| Measured live during the run | Analytical / from public benchmarks |
|---|---|
| CIFAR-10 / CIFAR-100 training accuracy and loss | Wireless uplink delay (`Exp(200 ms)`) |
| Per-round local-compute wall-clock | Edge aggregation delay (`Exp(150 ms)`) |
| Per-round n_active and dropout trace | Blockchain validation delay (`Exp(300 ms)`) |
| HMM forward / Baum-Welch (when fitting from observations) | PQ cycle counts (Kyber, Dilithium, Falcon, ...) � from public OQS benchmarks |
| DPP action selection + virtual-queue update | M/G/1 / M/M/c closed-form bounds |

Default parameter assumptions are printed by the notebook in cell 6 so
you can see them at a glance.

## Pro-tier Colab GPUs

The notebook benefits from any of these:

| GPU | Tier | CIFAR-100 full run (50 clients � 30 rounds) |
|---|---|---|
| A100 | Colab Pro+ | ~5 min |
| V100 | Colab Pro | ~8 min |
| T4 | Free / Pro | ~25�30 min |
| L4 | Pro | ~12 min |

Pick any of them under **Runtime ? Change runtime type ? GPU**. The
notebook auto-detects which one is attached and prints it in cell 4.
