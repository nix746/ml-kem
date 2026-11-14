"""Tests for the NTT and for vector/matrix operations in the NTT domain (FIPS 203, Section 4.3)."""

import numpy as np
import pytest
from conftest import KYBER_RING, from_kyber, random_poly, to_kyber, to_kyber_ntt

from params import N, Q
from polynomial import (
    GAMMAS,
    ZETAS,
    base_case_multiply,
    dot,
    inv_ntt,
    inv_ntt_vec,
    matvec,
    multiply_ntts,
    ntt,
    ntt_vec,
    poly_add,
    schoolbook_mul,
    transpose,
)

# --- ZETAS and GAMMAS ---


def test_zetas_match_fips_appendix_a():
    # First row of the ZETAS table in FIPS 203, Appendix A.
    assert ZETAS[:8] == [1, 1729, 2580, 3289, 2642, 630, 1897, 848]
    assert len(ZETAS) == 128


def test_gammas_match_fips_appendix_a():
    # FIPS 203, Appendix A lists 17, -17, 2761, -2761, 583, -583, 2649, -2649 (here mod Q).
    assert GAMMAS[:8] == [17, Q - 17, 2761, Q - 2761, 583, Q - 583, 2649, Q - 2649]
    assert len(GAMMAS) == 128


def test_zetas_match_kyber_py():
    assert ZETAS == KYBER_RING.ntt_zetas


def test_gammas_are_pairs_of_opposite_roots():
    # X^256 + 1 splits into (X^2 - g)(X^2 + g), so GAMMAS come in pairs g, -g.
    for i in range(0, 128, 2):
        assert (GAMMAS[i] + GAMMAS[i + 1]) % Q == 0


# --- ntt / inv_ntt ---


@pytest.mark.parametrize("seed", range(10))
def test_inv_ntt_undoes_ntt(seed):
    f = random_poly(np.random.default_rng(seed))

    assert np.array_equal(inv_ntt(ntt(f)), f)
    assert np.array_equal(ntt(inv_ntt(f)), f)


@pytest.mark.parametrize("seed", range(10))
def test_ntt_matches_kyber_py(seed):
    f = random_poly(np.random.default_rng(seed))

    assert np.array_equal(ntt(f), from_kyber(to_kyber(f).to_ntt()))
    assert np.array_equal(inv_ntt(f), from_kyber(to_kyber_ntt(f).from_ntt()))


@pytest.mark.parametrize("value", [0, 1, Q - 1])
def test_ntt_edge_cases(value):
    f = np.full(N, value, dtype=np.int64)

    assert np.array_equal(inv_ntt(ntt(f)), f)
    assert np.array_equal(ntt(f), from_kyber(to_kyber(f).to_ntt()))


def test_ntt_of_zero_is_zero():
    assert np.array_equal(ntt(np.zeros(N)), np.zeros(N))


def test_ntt_does_not_modify_input(rng):
    f = random_poly(rng)
    f_before = f.copy()

    ntt(f)
    inv_ntt(f)

    assert np.array_equal(f, f_before)


def test_ntt_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ntt(np.zeros(N - 1))
    with pytest.raises(ValueError):
        inv_ntt(np.zeros((2, N)))


# --- base_case_multiply / multiply_ntts ---


def test_base_case_multiply_small_example():
    # (1 + 2X)(3 + 4X) = 3 + 10X + 8X^2, and X^2 = gamma, so with gamma = 5: (3 + 40) + 10X.
    assert base_case_multiply(1, 2, 3, 4, 5) == (43, 10)


@pytest.mark.parametrize("seed", range(10))
def test_multiply_ntts_equals_schoolbook(seed):
    rng = np.random.default_rng(seed)
    f, g = random_poly(rng), random_poly(rng)

    product = inv_ntt(multiply_ntts(ntt(f), ntt(g)))

    assert np.array_equal(product, schoolbook_mul(f, g))


def test_multiply_ntts_with_all_q_minus_one():
    f = np.full(N, Q - 1, dtype=np.int64)

    assert np.array_equal(inv_ntt(multiply_ntts(ntt(f), ntt(f))), schoolbook_mul(f, f))


def test_multiply_ntts_matches_kyber_py(rng):
    f_hat, g_hat = random_poly(rng), random_poly(rng)

    expected = from_kyber(to_kyber_ntt(f_hat) * to_kyber_ntt(g_hat))
    assert np.array_equal(multiply_ntts(f_hat, g_hat), expected)


# --- vectors and matrices ---


def test_ntt_vec_and_inv_ntt_vec(rng):
    v = rng.integers(0, Q, size=(3, N), dtype=np.int64)

    v_hat = ntt_vec(v)

    assert v_hat.shape == (3, N)
    for i in range(3):
        assert np.array_equal(v_hat[i], ntt(v[i]))
    assert np.array_equal(inv_ntt_vec(v_hat), v)


def test_transpose(rng):
    a = rng.integers(0, Q, size=(3, 3, N), dtype=np.int64)

    a_t = transpose(a)

    for i in range(3):
        for j in range(3):
            assert np.array_equal(a_t[i, j], a[j, i])


@pytest.mark.parametrize("k", [2, 3, 4])
def test_dot_equals_schoolbook_sum(k, rng):
    u = rng.integers(0, Q, size=(k, N), dtype=np.int64)
    v = rng.integers(0, Q, size=(k, N), dtype=np.int64)

    expected = np.zeros(N, dtype=np.int64)
    for i in range(k):
        expected = poly_add(expected, schoolbook_mul(u[i], v[i]))

    assert np.array_equal(inv_ntt(dot(ntt_vec(u), ntt_vec(v))), expected)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_matvec_equals_schoolbook(k, rng):
    a_hat = rng.integers(0, Q, size=(k, k, N), dtype=np.int64)
    s = rng.integers(0, Q, size=(k, N), dtype=np.int64)
    a = np.array([inv_ntt_vec(row) for row in a_hat])

    expected = np.zeros((k, N), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            expected[i] = poly_add(expected[i], schoolbook_mul(a[i, j], s[j]))

    result = inv_ntt_vec(matvec(a_hat, ntt_vec(s)))

    assert result.shape == (k, N)
    assert np.array_equal(result, expected)


def test_matvec_with_transpose_uses_columns(rng):
    a_hat = rng.integers(0, Q, size=(2, 2, N), dtype=np.int64)
    y_hat = rng.integers(0, Q, size=(2, N), dtype=np.int64)

    result = matvec(transpose(a_hat), y_hat)

    # Row 0 of A^T is column 0 of A.
    expected0 = poly_add(multiply_ntts(a_hat[0, 0], y_hat[0]), multiply_ntts(a_hat[1, 0], y_hat[1]))
    assert np.array_equal(result[0], expected0)


def test_dot_rejects_different_lengths():
    with pytest.raises(ValueError):
        dot(np.zeros((2, N)), np.zeros((3, N)))
