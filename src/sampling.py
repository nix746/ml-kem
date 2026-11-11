"""Polynomial sampling: SampleNTT and SamplePolyCBD (FIPS 203, Section 4.2.2)."""

import numpy as np

from compression import bytes_to_bits
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

    FIPS 203, Algorithm 8. The input is 64 * eta bytes = 256 * 2 * eta bits. 
    Coefficient i uses the 2 * eta bits b[2 * eta * i : 2 * eta * (i + 1)]:
        f_i = (sum of the first eta bits) - (sum of the next eta bits) mod Q
    so f_i lies in [-eta, eta] (stored as Q - e for negative values), with mean 0 and variance eta / 2.
    """
    if eta not in (2, 3):
        raise ValueError(f"eta must be 2 or 3, got {eta}")
    if len(buf) != 64 * eta:
        raise ValueError(f"sample_poly_cbd with eta={eta} expects {64 * eta} bytes, got {len(buf)}")

    bits = bytes_to_bits(buf).reshape(N, 2 * eta)
    x = bits[:, :eta].sum(axis=1)
    y = bits[:, eta:].sum(axis=1)
    return (x - y) % Q
