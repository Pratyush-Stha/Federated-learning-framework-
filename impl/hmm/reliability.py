"""Hidden-Markov client reliability layer.

Implements the four-state model {Online, Busy, Offline, Byzantine} from
Section III-C of the paper. Two operations matter:

  1. forward()   -> returns rho_i(t) = P[ Z_i(t) in {Offline, Byzantine} ]
                    given the observation history.
  2. fit()       -> Baum-Welch EM on a batch of observation sequences.

We deliberately keep the implementation in NumPy (no hmmlearn dependency
during inference) so it can be embedded inside the FL server's hot path.
hmmlearn is only used when fit_hmmlearn() is called for a more robust EM.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

ONLINE, BUSY, OFFLINE, BYZ = 0, 1, 2, 3
N_STATES = 4
DROPOUT_MASK = np.array([0.0, 0.0, 1.0, 1.0])  # Offline + Byzantine count as drop


@dataclass
class HMMParams:
    """Per-client HMM: transition P (4x4), emission B (4x|O|), prior pi (4,)."""
    P: np.ndarray
    B: np.ndarray
    pi: np.ndarray

    @classmethod
    def reasonable_default(cls, n_obs: int = 3) -> "HMMParams":
        # Sticky transitions: clients tend to stay in their current state.
        P = np.array([
            [0.85, 0.08, 0.05, 0.02],
            [0.20, 0.70, 0.08, 0.02],
            [0.15, 0.05, 0.78, 0.02],
            [0.02, 0.03, 0.05, 0.90],
        ])
        # Emission: 0=ok-update, 1=missing, 2=deviation-flag.
        B = np.array([
            [0.90, 0.05, 0.05],
            [0.50, 0.45, 0.05],
            [0.00, 1.00, 0.00],
            [0.20, 0.10, 0.70],
        ])
        if n_obs != 3:
            B = np.ones((N_STATES, n_obs)) / n_obs
        pi = np.array([0.7, 0.2, 0.08, 0.02])
        return cls(P=P, B=B, pi=pi)


def forward(obs: np.ndarray, hp: HMMParams) -> np.ndarray:
    """Forward algorithm: returns alpha[t, k] = P(O_{1:t}, Z_t=k)."""
    T = len(obs)
    alpha = np.zeros((T, N_STATES))
    alpha[0] = hp.pi * hp.B[:, obs[0]]
    s = alpha[0].sum()
    if s > 0:
        alpha[0] /= s  # rescale to prevent underflow; this turns alpha into filtered posterior.
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ hp.P) * hp.B[:, obs[t]]
        s = alpha[t].sum()
        if s > 0:
            alpha[t] /= s
    return alpha


def dropout_probability(obs: np.ndarray, hp: HMMParams) -> float:
    """rho_i(t) used by the RL scheduler in eq. (Expected Participation)."""
    alpha = forward(obs, hp)
    return float(alpha[-1] @ DROPOUT_MASK)


def expected_participation(obs_per_client: list[np.ndarray], hp: HMMParams) -> float:
    """bar n(t) / N from the paper."""
    return 1.0 - np.mean([dropout_probability(o, hp) for o in obs_per_client])


def fit_baum_welch(
    sequences: list[np.ndarray],
    n_obs: int,
    n_iter: int = 50,
    tol: float = 1e-4,
    seed: int = 0,
) -> HMMParams:
    """Lightweight EM for the four-state HMM. sequences is a list of
    observation arrays (one per client), each containing integers in
    [0, n_obs)."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(N_STATES), size=N_STATES)
    B = rng.dirichlet(np.ones(n_obs), size=N_STATES)
    pi = rng.dirichlet(np.ones(N_STATES))

    prev_ll = -np.inf
    for it in range(n_iter):
        # Sufficient statistics
        gamma_sum = np.zeros(N_STATES)
        xi_sum = np.zeros((N_STATES, N_STATES))
        emit_sum = np.zeros((N_STATES, n_obs))
        pi_sum = np.zeros(N_STATES)
        ll = 0.0

        for obs in sequences:
            T = len(obs)
            if T < 2:
                continue
            alpha = np.zeros((T, N_STATES))
            beta = np.zeros((T, N_STATES))
            scale = np.zeros(T)
            alpha[0] = pi * B[:, obs[0]]
            scale[0] = alpha[0].sum() or 1.0
            alpha[0] /= scale[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ P) * B[:, obs[t]]
                scale[t] = alpha[t].sum() or 1.0
                alpha[t] /= scale[t]
            beta[T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = P @ (B[:, obs[t + 1]] * beta[t + 1])
                beta[t] /= scale[t + 1]

            ll += np.log(scale + 1e-300).sum()
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            for t in range(T - 1):
                xi = (alpha[t][:, None] * P) * (B[:, obs[t + 1]] * beta[t + 1])[None, :]
                xi /= xi.sum() + 1e-300
                xi_sum += xi
            gamma_sum += gamma[:-1].sum(axis=0)
            for t, o in enumerate(obs):
                emit_sum[:, o] += gamma[t]
            pi_sum += gamma[0]

        # M-step
        P = xi_sum / (gamma_sum[:, None] + 1e-300)
        B = emit_sum / (emit_sum.sum(axis=1, keepdims=True) + 1e-300)
        pi = pi_sum / (pi_sum.sum() + 1e-300)

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return HMMParams(P=P, B=B, pi=pi)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    true = HMMParams.reasonable_default()
    # Simulate 50 clients each with 30 rounds of observations.
    seqs = []
    for _ in range(50):
        T = 30
        obs = np.zeros(T, dtype=int)
        z = rng.choice(N_STATES, p=true.pi)
        for t in range(T):
            obs[t] = rng.choice(true.B.shape[1], p=true.B[z])
            z = rng.choice(N_STATES, p=true.P[z])
        seqs.append(obs)

    rho = np.mean([dropout_probability(o, true) for o in seqs])
    print(f"Mean rho_i(t) over 50 clients : {rho:.3f}")
    print(f"Expected participation p(t)    : {1 - rho:.3f}")

    fitted = fit_baum_welch(seqs, n_obs=3, n_iter=40, seed=1)
    print(f"Recovered transition diag      : {np.round(np.diag(fitted.P), 3)}")
    print(f"True       transition diag     : {np.round(np.diag(true.P), 3)}")
