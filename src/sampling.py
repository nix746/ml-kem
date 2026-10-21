"""Polynomial sampling: SampleNTT and SamplePolyCBD (FIPS 203, Section 4.2.2)."""

import numpy as np


def sample_ntt(rho: bytes, i: int, j: int) -> np.ndarray:
    """Samples a uniform polynomial in the NTT domain from seed rho and indices (i, j).

    FIPS 203, Algorithm 7.
    """
    raise NotImplementedError("TODO: Implement Algorithm 7 (SampleNTT) from FIPS 203.")


def sample_poly_cbd(eta: int, buf: bytes) -> np.ndarray:
    """Samples a noise polynomial from the centered binomial distribution CBD_eta.

    FIPS 203, Algorithm 8.
    """
    raise NotImplementedError("TODO: Implement Algorithm 8 (SamplePolyCBD) from FIPS 203.")
