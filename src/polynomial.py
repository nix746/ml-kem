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


ZETA = 17  # primitive 256-th root of unity mod Q (FIPS 203, Section 4.3)


def _bitrev7(i: int) -> int:
    """Reverses the 7 bits of i (0 <= i < 128)."""
    return int(format(i, "07b")[::-1], 2)


# ZETAS[i] = 17^BitRev7(i) mod Q: twiddle factors of the NTT (FIPS 203, Appendix A).
ZETAS = [pow(ZETA, _bitrev7(i), Q) for i in range(128)]
# GAMMAS[i] = 17^(2*BitRev7(i)+1) mod Q: the roots of X^2 - gamma, used in MultiplyNTTs.
GAMMAS = [pow(ZETA, 2 * _bitrev7(i) + 1, Q) for i in range(128)]
N_INV = 3303  # 128^-1 mod Q (not 256^-1: the NTT stops after 7 layers, pairs remain)


def _as_poly(f: np.ndarray) -> np.ndarray:
    """Copies f to int64 and checks that it is a single polynomial of shape (N,)."""
    f = np.array(f, dtype=np.int64)
    if f.shape != (N,):
        raise ValueError(f"expected a single polynomial of shape ({N},), got {f.shape}")
    return f


def ntt(f: np.ndarray) -> np.ndarray:
    """Number-Theoretic Transform (FIPS 203, Algorithm 9).

    Seven layers of butterflies (len = 128, 64, ..., 2). The result is 128 pairs, every pair
    being f mod (X^2 - gamma_i). The input is not modified.
    """
    f = _as_poly(f)
    k = 1
    length = 128
    while length >= 2:
        for start in range(0, N, 2 * length):
            zeta = ZETAS[k]
            k += 1
            for j in range(start, start + length):
                t = (zeta * f[j + length]) % Q
                f[j + length] = (f[j] - t) % Q
                f[j] = (f[j] + t) % Q
        length //= 2
    return f


def inv_ntt(f_hat: np.ndarray) -> np.ndarray:
    """Inverse Number-Theoretic Transform NTT^-1 (FIPS 203, Algorithm 10).

    Same layers as `ntt` in reverse order (len = 2, 4, ..., 128), k going down from 127,
    and a final multiplication by 128^-1 = 3303.
    """
    f = _as_poly(f_hat)
    k = 127
    length = 2
    while length <= 128:
        for start in range(0, N, 2 * length):
            zeta = ZETAS[k]
            k -= 1
            for j in range(start, start + length):
                t = f[j]
                f[j] = (t + f[j + length]) % Q
                f[j + length] = (zeta * (f[j + length] - t)) % Q
        length *= 2
    return (f * N_INV) % Q


def base_case_multiply(a0: int, a1: int, b0: int, b1: int, gamma: int) -> tuple[int, int]:
    """Product of (a0 + a1 X) and (b0 + b1 X) modulo (X^2 - gamma) (FIPS 203, Algorithm 12)."""
    c0 = (a0 * b0 + a1 * b1 * gamma) % Q
    c1 = (a0 * b1 + a1 * b0) % Q
    return c0, c1


def multiply_ntts(f_hat: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
    """Multiplication of two NTT representations (FIPS 203, Algorithm 11).

    Pair i (coefficients 2i and 2i+1), i = 0..127, is multiplied modulo X^2 - GAMMAS[i].
    """
    f_hat, g_hat = _as_int64_pair(f_hat, g_hat)
    if f_hat.ndim != 1:
        raise ValueError(f"expected a single polynomial of shape ({N},), got {f_hat.shape}")
    h = np.zeros(N, dtype=np.int64)
    for i in range(N // 2):
        gamma = GAMMAS[i]
        h[2 * i], h[2 * i + 1] = base_case_multiply(
            int(f_hat[2 * i]),
            int(f_hat[2 * i + 1]),
            int(g_hat[2 * i]),
            int(g_hat[2 * i + 1]),
            gamma,
        )
    return h
