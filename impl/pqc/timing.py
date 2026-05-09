"""Empirical timing for the post-quantum primitives in Table I.

Tries to load liboqs-python; if unavailable, falls back to a deterministic
stub model whose mean cycle counts come from published OQS benchmarks.
The stub keeps the rest of the pipeline reproducible without native deps.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

# CPU cycles -> seconds at this clock rate (used in eq. (pq_service)).
DEFAULT_CPU_GHZ = 3.2

# Order-of-magnitude cycle counts from the OQS reference implementations
# (used only as a stub when liboqs is not available; do not rely on for
# publishable numbers, run the real benchmark on the target hardware).
STUB_CYCLES = {
    "kyber512":              30_000,
    "ntru-hps-2048-509":     45_000,
    "frodokem-640-aes":     900_000,
    "dilithium2":           250_000,
    "falcon-512":         3_500_000,
}


@dataclass
class PQTiming:
    algo: str
    op: str        # "keypair" | "encaps" | "decaps" | "sign" | "verify"
    samples: list[float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def p95(self) -> float:
        return float(sorted(self.samples)[int(0.95 * len(self.samples))]) if self.samples else 0.0


def _measure_liboqs(algo: str, n: int = 100) -> PQTiming | None:
    """Real measurement via liboqs-python."""
    try:
        import oqs  # type: ignore
    except Exception:
        return None

    if algo in oqs.get_enabled_kem_mechanisms():
        kem = oqs.KeyEncapsulation(algo)
        pk = kem.generate_keypair()
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            ct, ss = kem.encap_secret(pk)
            samples.append(time.perf_counter() - t0)
        kem.free()
        return PQTiming(algo=algo, op="encaps", samples=samples)

    if algo in oqs.get_enabled_sig_mechanisms():
        sig = oqs.Signature(algo)
        pk = sig.generate_keypair()
        msg = b"qrqfl-bench"
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            sig.sign(msg)
            samples.append(time.perf_counter() - t0)
        sig.free()
        return PQTiming(algo=algo, op="sign", samples=samples)

    return None


def _measure_stub(algo: str, n: int = 100, cpu_ghz: float = DEFAULT_CPU_GHZ) -> PQTiming:
    cycles = STUB_CYCLES.get(algo.lower())
    if cycles is None:
        raise ValueError(f"Unknown algo for stub: {algo}")
    mean = cycles / (cpu_ghz * 1e9)
    # Add ~5% jitter so the queue model's S^2 moment is non-degenerate.
    samples = [mean * (1.0 + 0.05 * ((i % 7) - 3) / 3) for i in range(n)]
    return PQTiming(algo=algo, op="stub", samples=samples)


def measure(algo: str, n: int = 100) -> PQTiming:
    real = _measure_liboqs(algo, n)
    if real is not None and real.samples:
        return real
    return _measure_stub(algo, n)


if __name__ == "__main__":
    print(f"{'algo':<22}{'op':<8}{'mean (s)':>14}{'p95 (s)':>14}")
    for algo in STUB_CYCLES:
        t = measure(algo, n=200)
        print(f"{t.algo:<22}{t.op:<8}{t.mean:>14.6f}{t.p95:>14.6f}")
