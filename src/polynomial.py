"""Arithmetic in the polynomial ring R_q = Z_q[X]/(X^256 + 1) and the NTT."""

import numpy as np


def poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Coefficient-wise addition of two polynomials modulo Q."""
    raise NotImplementedError("TODO: Implement polynomial addition in R_q.")


def poly_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Coefficient-wise subtraction of two polynomials modulo Q."""
    raise NotImplementedError("TODO: Implement polynomial subtraction in R_q.")


def ntt(f: np.ndarray) -> np.ndarray:
    """Number-Theoretic Transform (FIPS 203, Algorithm 9)."""
    raise NotImplementedError("TODO: Implement Algorithm 9 (NTT) from FIPS 203.")


def inv_ntt(f_hat: np.ndarray) -> np.ndarray:
    """Inverse Number-Theoretic Transform NTT^-1 (FIPS 203, Algorithm 10)."""
    raise NotImplementedError("TODO: Implement Algorithm 10 (NTT^-1) from FIPS 203.")


def multiply_ntts(f_hat: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
    """Multiplication of two NTT representations (FIPS 203, Algorithm 11)."""
    raise NotImplementedError("TODO: Implement Algorithm 11 (MultiplyNTTs) from FIPS 203.")
