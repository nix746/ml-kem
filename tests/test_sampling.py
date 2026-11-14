"""Tests for SampleNTT and SamplePolyCBD (FIPS 203, Algorithms 7 and 8)."""

import numpy as np
import pytest
from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024

import sampling
from compress import bits_to_bytes
from params import N, Q
from sampling import sample_ntt, sample_poly_cbd

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


# SampleNTT (FIPS 203, Algorithm 7)
KYBER_BY_K = {2: ML_KEM_512, 3: ML_KEM_768, 4: ML_KEM_1024}


def our_matrix(rho, k):
    """A_hat built like in K-PKE: entry [i, j] = sample_ntt(rho, i, j). Shape (k, k, 256)."""
    return np.array([[sample_ntt(rho, i, j) for j in range(k)] for i in range(k)])


def kyber_matrix(rho, k, transpose=False):
    """The same matrix from kyber-py, converted to a NumPy array of shape (k, k, 256)."""
    a_hat = KYBER_BY_K[k]._generate_matrix_from_seed(rho, transpose=transpose)
    return np.array([[a_hat[i, j].coeffs for j in range(k)] for i in range(k)])


@pytest.mark.parametrize("k", [2, 3, 4])
def test_sample_ntt_full_matrix_matches_kyber_py(k):
    # The whole matrix, for ML-KEM-512, 768 and 1024. This is the test that catches a swap of
    # i and j: a swapped implementation still decrypts its own ciphertexts (it consistently
    # uses the transpose), so only a comparison with an independent implementation finds it.
    rng = np.random.default_rng(k)
    for _ in range(5):
        rho = rng.bytes(32)
        assert np.array_equal(our_matrix(rho, k), kyber_matrix(rho, k))


def test_sample_ntt_swapped_indices_give_the_transpose():
    rho = bytes(range(32))
    ours = our_matrix(rho, 3)
    transposed = kyber_matrix(rho, 3, transpose=True)
    assert not np.array_equal(ours, transposed)  # A_hat is not symmetric...
    assert np.array_equal(ours.transpose(1, 0, 2), transposed)  # ...so i, j order matters


def test_sample_ntt_output_shape_and_range():
    rng = np.random.default_rng(7)
    for _ in range(20):
        f = sample_ntt(rng.bytes(32), 1, 2)
        assert f.shape == (N,)
        assert f.dtype == np.int64
        assert f.min() >= 0
        assert f.max() < Q


def test_sample_ntt_is_roughly_uniform():
    # Uniform on [0, Q): mean (Q - 1) / 2 = 1664 and every eighth of [0, Q) gets ~1/8 of values.
    rng = np.random.default_rng(8)
    samples = np.concatenate([sample_ntt(rng.bytes(32), 0, 0) for _ in range(200)])
    assert abs(samples.mean() - (Q - 1) / 2) < 15
    buckets = np.bincount(samples * 8 // Q, minlength=8) / samples.size
    assert np.all(np.abs(buckets - 1 / 8) < 0.01)


def test_sample_ntt_requests_more_bytes_when_needed(monkeypatch):
    # Fake XOF: the first 840 bytes are 0xFF, so every candidate is 4095 >= Q and is rejected;
    # after that the stream is all zeros (always accepted). sample_ntt must ask for more bytes.
    def fake_xof(seed, length):
        stream = b"\xff" * 840 + bytes(10_000)
        return stream[:length]

    monkeypatch.setattr(sampling, "xof", fake_xof)
    assert not sample_ntt(bytes(32), 0, 0).any()


def test_sample_ntt_rejects_bad_input():
    with pytest.raises(ValueError):
        sample_ntt(bytes(31), 0, 0)  # rho must be 32 bytes
    with pytest.raises(ValueError):
        sample_ntt(bytes(32), 256, 0)  # i and j are single bytes
