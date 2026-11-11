"""Tests for SamplePolyCBD (FIPS 203, Algorithm 8)."""

import numpy as np
import pytest
from kyber_py.ml_kem import ML_KEM_768

from compress import bits_to_bytes
from params import Q
from sampling import sample_poly_cbd

ETA_VALUES = [2, 3]


def centered(f):
    """Maps values from [0, Q) to the centered range (-Q/2, Q/2]: e.g. 3328 -> -1."""
    f = np.asarray(f)
    return np.where(f > Q // 2, f - Q, f)


# Comparison with the reference oracle (kyber-py)


@pytest.mark.parametrize("eta", ETA_VALUES)
def test_cbd_matches_kyber_py(eta):
    rng = np.random.default_rng(eta)
    for _ in range(50):
        buf = rng.bytes(64 * eta)
        expected = ML_KEM_768.R.cbd(buf, eta).coeffs
        assert sample_poly_cbd(eta, buf).tolist() == expected


# Examples by hand


def test_cbd_example_by_hand_eta2():
    # eta = 2: one byte gives two coefficients (4 bits each, least significant bit first).
    buf = bytearray(128)
    buf[0] = 0b00001011  # bits [1,1,0,1, 0,0,0,0]: f_0 = (1+1) - (0+1) = 1, f_1 = 0 - 0 = 0
    buf[1] = 0b11000000  # bits [0,0,0,0, 0,0,1,1]: f_2 = 0 - 0 = 0,       f_3 = 0 - 2 = -2
    f = sample_poly_cbd(2, bytes(buf))
    assert f[:4].tolist() == [1, 0, 0, Q - 2]


def test_cbd_all_zero_bytes_give_zero_polynomial():
    for eta in ETA_VALUES:
        assert not sample_poly_cbd(eta, bytes(64 * eta)).any()


# Mathematical properties


@pytest.mark.parametrize("eta", ETA_VALUES)
def test_cbd_values_are_small(eta):
    rng = np.random.default_rng(10 + eta)
    for _ in range(50):
        f = centered(sample_poly_cbd(eta, rng.bytes(64 * eta)))
        assert f.min() >= -eta
        assert f.max() <= eta


def all_bit_patterns_buffer(eta):
    """A buffer in which every possible group of 2*eta bits occurs equally often.

    There are 2^(2*eta) patterns (16 for eta = 2, 64 for eta = 3) and 256 coefficients, so each
    pattern is used 256 / 2^(2*eta) times.
    """
    width = 2 * eta
    patterns = np.arange(1 << width)  # 0, 1, ..., 2^width - 1
    repeats = 256 // len(patterns)
    groups = np.tile(patterns, repeats)  # 256 patterns, one per coefficient
    bits = (groups[:, None] >> np.arange(width)) & 1  # shape (256, width)
    return bits_to_bytes(bits.reshape(-1))


# Exact CBD_eta probabilities times 2^(2*eta), for the values -eta, ..., eta.
# eta = 2: 1 : 4 : 6 : 4 : 1 (out of 16);  eta = 3: 1 : 6 : 15 : 20 : 15 : 6 : 1 (out of 64).
EXACT_COUNTS = {2: [1, 4, 6, 4, 1], 3: [1, 6, 15, 20, 15, 6, 1]}


@pytest.mark.parametrize("eta", ETA_VALUES)
def test_cbd_has_exact_binomial_distribution(eta):
    # Feeding every bit pattern equally often must reproduce the CBD probabilities exactly.
    f = centered(sample_poly_cbd(eta, all_bit_patterns_buffer(eta)))
    counts = [int(np.sum(f == v)) for v in range(-eta, eta + 1)]
    repeats = 256 // (1 << (2 * eta))
    assert counts == [c * repeats for c in EXACT_COUNTS[eta]]


@pytest.mark.parametrize("eta", ETA_VALUES)
def test_cbd_mean_and_variance(eta):
    # On random input: mean close to 0 and variance close to eta / 2.
    rng = np.random.default_rng(100 + eta)
    samples = np.concatenate(
        [centered(sample_poly_cbd(eta, rng.bytes(64 * eta))) for _ in range(400)]
    )  # 400 * 256 = 102400 coefficients
    assert abs(samples.mean()) < 0.02
    assert abs(samples.var() - eta / 2) < 0.03


# Input validation


@pytest.mark.parametrize("bad_eta", [0, 1, 4])
def test_cbd_rejects_invalid_eta(bad_eta):
    with pytest.raises(ValueError):
        sample_poly_cbd(bad_eta, bytes(64 * 2))


@pytest.mark.parametrize("eta", ETA_VALUES)
def test_cbd_rejects_wrong_length(eta):
    with pytest.raises(ValueError):
        sample_poly_cbd(eta, bytes(64 * eta - 1))
    with pytest.raises(ValueError):
        sample_poly_cbd(eta, bytes(64 * eta + 1))
