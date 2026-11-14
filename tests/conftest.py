"""Shared fixtures and adapters between our NumPy arrays and kyber-py objects."""

import numpy as np
import pytest
from kyber_py.polynomials.polynomials import PolynomialRing

from params import N, Q

KYBER_RING = PolynomialRing()


def to_kyber(poly: np.ndarray):
    """Our polynomial (shape (N,)) -> kyber-py polynomial in the normal domain."""
    return KYBER_RING([int(x) for x in poly])


def to_kyber_ntt(poly_hat: np.ndarray):
    """Our polynomial in the NTT domain (shape (N,)) -> kyber-py polynomial in the NTT domain."""
    return KYBER_RING([int(x) for x in poly_hat], is_ntt=True)


def from_kyber(poly) -> np.ndarray:
    """kyber-py polynomial (either domain) -> our polynomial of shape (N,)."""
    return np.array(poly.coeffs, dtype=np.int64)


@pytest.fixture
def rng() -> np.random.Generator:
    """Random generator with a fixed seed, so every test run uses the same data."""
    return np.random.default_rng(2024)


def random_poly(rng: np.random.Generator) -> np.ndarray:
    """Random polynomial with coefficients in [0, Q)."""
    return rng.integers(0, Q, size=N, dtype=np.int64)
