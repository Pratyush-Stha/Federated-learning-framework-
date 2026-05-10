# Refresh measured results for `main.tex`

The Results section of `main.tex` consumes auto-generated LaTeX snippets and
plots from `current works in progress/figures/auto/`. Those files are
overwritten by `code/impl/build_paper_tables.py` after a baseline run finishes.

If `figures/auto/` is missing or contains only placeholders, the paper still
compiles, but `tab:baselines_compare`, `tab:ablation_compare`,
`fig:accuracy_compare`, and `fig:final_acc_bar` will display "pending"
labels. Replace the placeholders by following the steps below.

## Methods that get measured

`code/impl/run_baselines.py` runs all of these under one Flower simulation,
sharing dataset, partition, model, batch size, seed, and round count:

| ID                  | Description                                |
| ------------------- | ------------------------------------------ |
| `fedavg`            | Vanilla FedAvg, random sampling, no PQ overhead |
| `fedavg_kyber`      | Vanilla FedAvg with fixed Kyber overhead   |
| `fedprox`           | FedAvg-style sampling + FedProx mu>0       |
| `qrqfl`             | Full QRQ-FL (HMM + DPP + adaptive PQ)      |
| `qrqfl_no_hmm`      | DPP only (HMM disabled)                    |
| `qrqfl_no_dpp`      | HMM only (DPP disabled, top-k by reliab.)  |
| `qrqfl_fixed_kyber` | HMM + DPP, PQ scheme forced to Kyber       |

## Step 1 - run all baselines (Colab recommended)

In Colab:

1. Open `code/impl/QRQFL_colab.ipynb`.
2. Set `RUN_FEDAVG_BASELINE = False` in section 3 if you do not also need
   the quick-path comparison (saves wall time).
3. Scroll to section 15 and set `RUN_BASELINE_GRID = True`.
4. Run the notebook end-to-end. Wall time on a T4: roughly 30-45 minutes
   for the default `cifar100`, `N=10`, `K=5`, 8 rounds preset. Use a
   higher GPU tier for larger configurations.
5. The grid writes a folder of the form
   `runs/baselines_YYYYMMDD_HHMMSS/` containing `summary.json`,
   `accuracy_curves.csv`, and per-method outputs.

Local invocation (`flwr[simulation]`, `ray`, `torch`, `torchvision`
already installed):

```
cd code/impl
python run_baselines.py --dataset cifar100 --clients 10 --select 5 --rounds 8
```

Add `--smoke` for a 1-round CPU sanity check.

## Step 2 - regenerate paper artefacts

```
cd code/impl
python build_paper_tables.py runs/baselines_YYYYMMDD_HHMMSS
```

This rewrites `current works in progress/figures/auto/`:

- `method_summary.tex` - one paragraph describing the run config.
- `baselines_table.tex` - measured baselines table.
- `ablation_table.tex` - QRQ-FL ablation table.
- `accuracy_compare.pdf/.png` - per-round accuracy lines.
- `final_accuracy_bar.pdf/.png` - bar chart of final accuracy.
- `PROVENANCE.txt` - which run was used to populate the snippets.

Recompile `main.tex`. The Results section now reflects the real run.

## Safety

- Do not edit files inside `figures/auto/` by hand; they are regenerated.
- If `summary.json` is missing fields, `build_paper_tables.py` will leave
  the corresponding cells as `--`, never as fabricated numbers.
- The current verified QRQ-FL vs FedAvg validation run remains the source
  of truth for the existing per-round table and accuracy figure in
  Section IV-C; the new baselines/ablations are appended in their own
  tables and can be regenerated independently.
