"""Synthetic IoV telemetry generator.

Produces a temporal regression task per virtual vehicle: predict next-step
speed and lane-change probability from (gps_x, gps_y, speed, accel, heading)
windows of length W. Heterogeneity across vehicles comes from per-vehicle
mean speed, road class, and noise scale, which mimics the non-IID profile
of a real fleet while staying in-tree (no downloads).
"""
from __future__ import annotations

import numpy as np


def generate(
    n_vehicles: int = 200,
    n_steps: int = 600,
    window: int = 20,
    seed: int = 0,
):
    """Returns dict[vehicle_id] = (X, y) with X shape (n_windows, W, 5)
    and y shape (n_windows, 2)."""
    rng = np.random.default_rng(seed)
    fleet = {}
    for v in range(n_vehicles):
        mean_speed = rng.uniform(8, 28)              # m/s, ~30..100 km/h
        noise = rng.uniform(0.2, 1.5)
        road_class = rng.choice(["urban", "highway"], p=[0.6, 0.4])
        heading = rng.uniform(0, 2 * np.pi)
        x = np.zeros(n_steps); y_ = np.zeros(n_steps)
        speed = np.zeros(n_steps); accel = np.zeros(n_steps)
        heading_t = np.full(n_steps, heading)
        speed[0] = mean_speed
        for t in range(1, n_steps):
            dv = rng.normal(0, noise)
            speed[t] = np.clip(speed[t - 1] + dv, 0, 35)
            accel[t] = speed[t] - speed[t - 1]
            heading_t[t] = heading_t[t - 1] + rng.normal(0, 0.02)
            x[t] = x[t - 1] + speed[t] * np.cos(heading_t[t])
            y_[t] = y_[t - 1] + speed[t] * np.sin(heading_t[t])

        feats = np.stack([x, y_, speed, accel, heading_t], axis=1).astype(np.float32)
        # supervised pairs: window -> (next_speed, lane_change_prob proxy)
        Xw, yw = [], []
        for t in range(window, n_steps - 1):
            Xw.append(feats[t - window : t])
            lane_change = 1.0 if abs(np.cos(heading_t[t]) - np.cos(heading_t[t - 1])) > 0.05 else 0.0
            yw.append([speed[t + 1], lane_change])
        fleet[v] = (np.stack(Xw), np.asarray(yw, dtype=np.float32))
    return fleet


if __name__ == "__main__":
    fleet = generate(n_vehicles=10, n_steps=200, window=20, seed=1)
    sizes = [v[0].shape[0] for v in fleet.values()]
    print(f"Generated {len(fleet)} vehicles, samples each = {sizes[0]} ({len(set(sizes))} distinct sizes)")
    v0_X, v0_y = fleet[0]
    print(f"  vehicle 0: X={v0_X.shape}, y={v0_y.shape}, y_speed mean={v0_y[:,0].mean():.2f}")
