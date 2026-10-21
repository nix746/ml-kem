"""Losowanie wielomianów: SampleNTT oraz SamplePolyCBD (FIPS 203, Sekcja 4.2.2)."""

import numpy as np


def sample_ntt(rho: bytes, i: int, j: int) -> np.ndarray:
    """Generuje losowy wielomian w domenie NTT z ziarna rho i indeksów (i, j) (Algorithm 7)."""
    raise NotImplementedError("TODO: Implementacja Algorithm 7 (SampleNTT) z FIPS 203.")


def sample_poly_cbd(eta: int, buf: bytes) -> np.ndarray:
    """Generuje wielomian szumu z wycentrowanego rozkładu dwumianowego CBD_eta (Algorithm 8)."""
    raise NotImplementedError("TODO: Implementacja Algorithm 8 (SamplePolyCBD) z FIPS 203.")
