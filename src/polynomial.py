"""Arithmetic in the polynomial ring R_q = Z_q[X]/(X^256 + 1) and the NTT."""

import numpy as np

from params import N, Q


def _as_int64_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts two same-shaped polynomials (or vectors of them) to int64 (CONTRACT, Section 2)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.shape != b.shape or a.shape[-1:] != (N,):
        raise ValueError(f"expected equal shapes ending in {N}, got {a.shape} and {b.shape}")
    return a, b


def poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Coefficient-wise addition of two polynomials modulo Q."""
    a, b = _as_int64_pair(a, b)
    return (a + b) % Q


def poly_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Coefficient-wise subtraction of two polynomials modulo Q."""
    a, b = _as_int64_pair(a, b)
    return (a - b) % Q


def schoolbook_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Schoolbook multiplication in R_q, with terms from X^256 up wrapped by X^256 = -1.

    A single coefficient of the full product sums up to 256 * (Q - 1)^2 = 2.8 * 10^9, which does
    not fit in int32 and would wrap around silently, so the sums are accumulated in int64.
    """
    a, b = _as_int64_pair(a, b)
    if a.ndim != 1:
        raise ValueError(f"expected a single polynomial of shape ({N},), got {a.shape}")
    c = np.zeros(2 * N, dtype=np.int64)
    for i in range(N):
        c[i : i + N] += a[i] * b
    return (c[:N] - c[N:]) % Q


def ntt(f: np.ndarray) -> np.ndarray:
    """Number-Theoretic Transform (FIPS 203, Algorithm 9)."""
    raise NotImplementedError("TODO: Implement Algorithm 9 (NTT) from FIPS 203.")


def inv_ntt(f_hat: np.ndarray) -> np.ndarray:
    """Inverse Number-Theoretic Transform NTT^-1 (FIPS 203, Algorithm 10)."""
    raise NotImplementedError("TODO: Implement Algorithm 10 (NTT^-1) from FIPS 203.")


def multiply_ntts(f_hat: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
    """Multiplication of two NTT representations (FIPS 203, Algorithm 11)."""
    raise NotImplementedError("TODO: Implement Algorithm 11 (MultiplyNTTs) from FIPS 203.")
