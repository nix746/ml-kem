"""Polynomial sampling: SampleNTT and SamplePolyCBD (FIPS 203, Section 4.2.2)."""

import numpy as np

from crypto_primitives import xof
from params import N, Q


def sample_ntt(rho: bytes, i: int, j: int) -> np.ndarray:
    """Samples a uniform polynomial in the NTT domain from seed rho and indices (i, j).

    FIPS 203, Algorithm 7. The XOF input is rho || j || i (j first!), so that A_hat[i, j] is
    sampled from `sample_ntt(rho, i, j)`;

    Rejection sampling: every 3 bytes (b0, b1, b2) give two 12-bit candidates
        d1 = b0 + 256 * (b1 mod 16),   d2 = b1 // 16 + 16 * b2,
    and a candidate is kept only if it is < Q (otherwise it would not be uniform mod Q).
    """
    if len(rho) != 32:
        raise ValueError(f"rho must be exactly 32 bytes, got {len(rho)}")
    if not (0 <= i <= 255 and 0 <= j <= 255):
        raise ValueError(f"i and j must be in [0, 255], got {i}, {j}")

    seed = rho + bytes([j, i])
    length = 840
    while True:
        # SHAKE128 output is a prefix-stable stream, so asking for more bytes = "squeeze more".
        data = np.frombuffer(xof(seed, length), dtype=np.uint8).astype(np.int64)
        triples = data[: len(data) // 3 * 3].reshape(-1, 3)
        d1 = triples[:, 0] + 256 * (triples[:, 1] % 16)
        d2 = triples[:, 1] // 16 + 16 * triples[:, 2]
        candidates = np.stack([d1, d2], axis=1).reshape(-1)
        accepted = candidates[candidates < Q]
        if len(accepted) >= N:
            return accepted[:N].copy()
        length *= 2


def sample_poly_cbd(eta: int, buf: bytes) -> np.ndarray:
    """Samples a noise polynomial from the centered binomial distribution CBD_eta.

    FIPS 203, Algorithm 8.
    """
    raise NotImplementedError("TODO: Implement Algorithm 8 (SamplePolyCBD) from FIPS 203.")
