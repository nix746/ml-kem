"""Arytmetyka w pierścieniu wielomianów R_q = Z_q[X]/(X^256 + 1) oraz transformata NTT."""

import numpy as np


def poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dodawanie dwóch wielomianów współczynnik po współczynniku modulo Q."""
    raise NotImplementedError("TODO: Implementacja dodawania wielomianów w R_q.")


def poly_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Odejmowanie dwóch wielomianów współczynnik po współczynniku modulo Q."""
    raise NotImplementedError("TODO: Implementacja odejmowania wielomianów w R_q.")


def ntt(f: np.ndarray) -> np.ndarray:
    """Transformata NTT (FIPS 203, Algorithm 9)."""
    raise NotImplementedError("TODO: Implementacja Algorithm 9 (NTT) z FIPS 203.")


def inv_ntt(f_hat: np.ndarray) -> np.ndarray:
    """Odwrotna transformata NTT^-1 (FIPS 203, Algorithm 10)."""
    raise NotImplementedError("TODO: Implementacja Algorithm 10 (NTT^-1) z FIPS 203.")


def multiply_ntts(f_hat: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
    """Mnożenie dwóch reprezentacji NTT (FIPS 203, Algorithm 11)."""
    raise NotImplementedError("TODO: Implementacja Algorithm 11 (MultiplyNTTs) z FIPS 203.")
