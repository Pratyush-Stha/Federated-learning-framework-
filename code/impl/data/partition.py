"""Non-IID partitioners shared by all datasets.

  - dirichlet_partition()  : label-skew via Dirichlet(alpha)  (CIFAR-100)
  - per_writer_partition() : returns the LEAF FEMNIST per-writer split
                             (call after running LEAF preprocess.sh)
  - hospital_partition()   : MIMIC-III hospital-as-client
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import numpy as np


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float = 0.3, seed: int = 0):
    """Assign sample indices to clients with label-Dirichlet skew.
    Returns: list[np.ndarray] of length n_clients."""
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for arr in idx_by_class:
        rng.shuffle(arr)
    splits = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        proportions = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(proportions) * len(idx_by_class[c])).astype(int)[:-1]
        for k, part in enumerate(np.split(idx_by_class[c], cuts)):
            splits[k].extend(part.tolist())
    return [np.array(sorted(s), dtype=np.int64) for s in splits]


def per_writer_partition(leaf_femnist_dir: str | pathlib.Path):
    """Read LEAF FEMNIST per-writer JSON files. Each file maps user_id ->
    {x: [...], y: [...]}. We just yield (user_id, X, y)."""
    base = pathlib.Path(leaf_femnist_dir)
    train = base / "data" / "train"
    if not train.exists():
        raise FileNotFoundError(
            f"FEMNIST not preprocessed yet: {train} missing. "
            "Run `cd leaf/data/femnist && ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample`"
        )
    for jf in sorted(train.glob("*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            blob = json.load(f)
        for uid in blob["users"]:
            ud = blob["user_data"][uid]
            yield uid, np.asarray(ud["x"], dtype=np.float32), np.asarray(ud["y"], dtype=np.int64)


def hospital_partition(features: np.ndarray, hospital_ids: np.ndarray):
    """Group MIMIC-III rows by hospital id."""
    parts = defaultdict(list)
    for i, h in enumerate(hospital_ids):
        parts[int(h)].append(i)
    return {h: np.array(idx, dtype=np.int64) for h, idx in parts.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake_labels = rng.integers(0, 100, size=50_000)
    splits = dirichlet_partition(fake_labels, n_clients=100, alpha=0.3, seed=0)
    sizes = [len(s) for s in splits]
    print(f"100-way Dirichlet(0.3) over 50k samples")
    print(f"  min/median/max client size = {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")
