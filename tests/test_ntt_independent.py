"""Independent checks of the NTT (FIPS 203, Section 4.3).

These tests do not reuse anything from polynomial.py except the functions under test: the
reference results are computed from the mathematical definitions (the factorization of
X^256 + 1, the Chinese Remainder map and the negacyclic product), so a mistake shared by
the implementation and its own tests would still be caught here.
"""

import numpy as np
import pytest

from params import N, Q
from polynomial import inv_ntt, multiply_ntts, ntt

ZETA = 17  # primitive 256-th root of unity modulo Q


def bitrev7(i):
    """Reverses the 7 bits of i (0 <= i < 128): e.g. 1 = 0000001 -> 1000000 = 64."""
    return int(f"{i:07b}"[::-1], 2)


GAMMA = [pow(ZETA, 2 * bitrev7(i) + 1, Q) for i in range(128)]


def negacyclic_mul(a, b):
    """a * b in Z_q[X]/(X^256 + 1) by definition: full product of degree <= 510, then X^256 = -1."""
    full = np.append(np.convolve(a.astype(np.int64), b.astype(np.int64)), 0)  # length 512
    return (full[:N] - full[N:]) % Q


def ntt_by_definition(f):
    """The NTT as the Chinese Remainder map: pair i of the result is f mod (X^2 - GAMMA[i]).

    In Z_q[X]/(X^2 - gamma) we have X^2 = gamma, so
        f(X) = sum_j f_{2j} X^{2j} + X * sum_j f_{2j+1} X^{2j}
             = sum_j f_{2j} gamma^j + X * sum_j f_{2j+1} gamma^j.
    """
    even, odd = f[0::2], f[1::2]
    out = np.zeros(N, dtype=np.int64)
    for i in range(128):
        powers = np.array([pow(GAMMA[i], j, Q) for j in range(128)])
        out[2 * i] = (even * powers % Q).sum() % Q
        out[2 * i + 1] = (odd * powers % Q).sum() % Q
    return out


def random_poly(seed):
    return np.random.default_rng(seed).integers(0, Q, size=N, dtype=np.int64)


def monomial(power):
    poly = np.zeros(N, dtype=np.int64)
    poly[power] = 1
    return poly


# The algebra behind the NTT
def test_zeta_is_a_primitive_256th_root_of_unity():
    assert pow(ZETA, 256, Q) == 1
    assert pow(ZETA, 128, Q) == Q - 1  # 17^128 = -1, so the order is exactly 256


def test_x256_plus_1_splits_into_128_quadratics():
    # X^256 + 1 = prod_i (X^2 - GAMMA[i]) mod Q. This factorization is why the NTT exists:
    # R_q is isomorphic to the product of the 128 small rings Z_q[X]/(X^2 - GAMMA[i]).
    product = [1]  # coefficients, lowest degree first
    for gamma in GAMMA:
        new = [0] * (len(product) + 2)
        for k, c in enumerate(product):
            new[k + 2] = (new[k + 2] + c) % Q  # c * X^2
            new[k] = (new[k] - c * gamma) % Q  # c * (-gamma)
        product = new
    expected = [1] + [0] * 255 + [1]  # 1 + X^256
    assert product == expected


# NTT against its definition
@pytest.mark.parametrize("seed", range(5))
def test_ntt_matches_chinese_remainder_definition(seed):
    f = random_poly(seed)
    assert np.array_equal(ntt(f), ntt_by_definition(f))


@pytest.mark.parametrize("power", [0, 1, 2, 127, 255])
def test_ntt_of_monomials(power):
    assert np.array_equal(ntt(monomial(power)), ntt_by_definition(monomial(power)))


def test_ntt_is_linear():
    a, b = random_poly(10), random_poly(11)
    assert np.array_equal(ntt((a + b) % Q), (ntt(a) + ntt(b)) % Q)
    assert np.array_equal(ntt(5 * a % Q), 5 * ntt(a) % Q)


# Multiplication through the NTT = ordinary multiplication
@pytest.mark.parametrize("seed", range(10))
def test_multiplication_through_ntt_equals_negacyclic_product(seed):
    a, b = random_poly(2 * seed), random_poly(2 * seed + 1)
    via_ntt = inv_ntt(multiply_ntts(ntt(a), ntt(b)))
    assert np.array_equal(via_ntt, negacyclic_mul(a, b))


def test_x_times_x255_is_minus_one_through_ntt():
    # X * X^255 = X^256 = -1 in R_q.
    via_ntt = inv_ntt(multiply_ntts(ntt(monomial(1)), ntt(monomial(255))))
    expected = np.zeros(N, dtype=np.int64)
    expected[0] = Q - 1
    assert np.array_equal(via_ntt, expected)


def test_small_noise_times_uniform_like_in_k_pke():
    # The typical K-PKE product: a uniform polynomial times a small CBD-like one.
    a = random_poly(20)
    s = np.random.default_rng(21).integers(-2, 3, size=N) % Q
    assert np.array_equal(inv_ntt(multiply_ntts(ntt(a), ntt(s))), negacyclic_mul(a, s))
