"""Offline smoke test for build_paper_tables.py.

Writes a small synthetic summary into a temporary directory, runs the table
builder, asserts the output files exist, and then deletes the temporary
directory.  *Does not* touch ``current works in progress/figures/auto`` -- the
test points the builder at a separate scratch directory by monkey-patching
``AUTO_DIR``.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import build_paper_tables  # type: ignore


def main() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="qrqfl_smoke_"))
    try:
        run_dir = tmp / "run"
        run_dir.mkdir()
        summary = {
            "config": {
                "dataset": "cifar100", "n_clients": 10, "n_select": 5,
                "n_rounds": 2, "batch_size": 32, "alpha": 0.3, "seed": 42,
                "device": "cpu", "pq_set": ["kyber512", "dilithium2", "falcon-512"],
                "pq_means_seconds": {"kyber512": 0.001, "dilithium2": 0.002,
                                       "falcon-512": 0.003},
            },
            "methods": [
                {"id": "fedavg", "label": "FedAvg", "kind": "fedavg",
                 "qrq": {}, "pq_force": None, "wall_seconds": 1.0,
                 "final_accuracy": 0.10, "rounds_run": 2},
                {"id": "fedavg_kyber", "label": "FedAvg + Kyber", "kind": "fedavg",
                 "qrq": {}, "pq_force": "kyber512", "wall_seconds": 1.1,
                 "final_accuracy": 0.11, "rounds_run": 2},
                {"id": "fedprox", "label": "FedProx", "kind": "qrqfl",
                 "qrq": {"use_hmm": False, "use_dpp": False, "proximal_mu": 0.01},
                 "pq_force": None, "wall_seconds": 1.2,
                 "final_accuracy": 0.12, "rounds_run": 2},
                {"id": "qrqfl", "label": "QRQ-FL (full)", "kind": "qrqfl",
                 "qrq": {"use_hmm": True, "use_dpp": True}, "pq_force": None,
                 "wall_seconds": 1.3, "final_accuracy": 0.15, "rounds_run": 2},
                {"id": "qrqfl_no_hmm", "label": "QRQ-FL w/o HMM", "kind": "qrqfl",
                 "qrq": {"use_hmm": False, "use_dpp": True}, "pq_force": None,
                 "wall_seconds": 1.0, "final_accuracy": 0.13, "rounds_run": 2},
                {"id": "qrqfl_no_dpp", "label": "QRQ-FL w/o DPP", "kind": "qrqfl",
                 "qrq": {"use_hmm": True, "use_dpp": False}, "pq_force": None,
                 "wall_seconds": 0.9, "final_accuracy": 0.14, "rounds_run": 2},
                {"id": "qrqfl_fixed_kyber", "label": "QRQ-FL fixed Kyber",
                 "kind": "qrqfl",
                 "qrq": {"use_hmm": True, "use_dpp": True, "fixed_pq": "kyber512"},
                 "pq_force": "kyber512", "wall_seconds": 1.05,
                 "final_accuracy": 0.145, "rounds_run": 2},
            ],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        rows = ["method_id,label,round,accuracy"]
        for m in summary["methods"]:
            for r in (1, 2):
                rows.append(f"{m['id']},{m['label']},{r},{0.05 * r + (0.01 * (m['id'] == 'qrqfl')):.4f}")
        (run_dir / "accuracy_curves.csv").write_text("\n".join(rows) + "\n")

        scratch_auto = tmp / "auto"
        build_paper_tables.AUTO_DIR = scratch_auto
        build_paper_tables.main([str(run_dir)])

        expected = [
            "method_summary.tex", "baselines_table.tex", "ablation_table.tex",
            "accuracy_compare.pdf", "accuracy_compare.png",
            "final_accuracy_bar.pdf", "final_accuracy_bar.png",
            "PROVENANCE.txt",
        ]
        missing = [e for e in expected if not (scratch_auto / e).exists()]
        if missing:
            raise SystemExit(f"Smoke test failed, missing: {missing}")
        # Spot-check the table contents (synthetic; just look for the labels).
        baseline_text = (scratch_auto / "baselines_table.tex").read_text()
        assert "FedAvg" in baseline_text and "QRQ-FL (full)" in baseline_text
        ablation_text = (scratch_auto / "ablation_table.tex").read_text()
        assert "QRQ-FL w/o HMM" in ablation_text
        print("Smoke test OK; outputs at", scratch_auto)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
