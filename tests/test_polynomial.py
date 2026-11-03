"""Tests for arithmetic in the polynomial ring R_q."""

import numpy as np
import pytest
from kyber_py.polynomials.polynomials_generic import GenericPolynomialRing

from params import N, Q
from polynomial import poly_add, poly_sub, schoolbook_mul

KYBER_RING = GenericPolynomialRing(Q, N)


def monomial(power: int, coefficient: int = 1) -> np.ndarray:
    poly = np.zeros(N, dtype=np.int64)
    poly[power] = coefficient
    return poly


def random_poly(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, Q, size=N, dtype=np.int64)


def convolve_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Independent reference for schoolbook_mul: full product via np.convolve, then the wrap."""
    c = np.append(np.convolve(a.astype(np.int64), b.astype(np.int64)), 0)
    return (c[:N] - c[N:]) % Q


def test_add_and_sub_stay_in_canonical_range():
    a = monomial(0, Q - 1)
    b = monomial(0, 2)

    assert poly_add(a, b)[0] == 1
    assert poly_sub(b, a)[0] == 3
    assert poly_sub(np.zeros(N, dtype=np.int64), monomial(0))[0] == Q - 1


def test_add_and_sub_are_inverse():
    rng = np.random.default_rng(1)
    a, b = random_poly(rng), random_poly(rng)

    assert np.array_equal(poly_sub(poly_add(a, b), b), a)


def test_add_and_sub_work_on_vectors_of_polynomials():
    rng = np.random.default_rng(2)
    a = rng.integers(0, Q, size=(3, N), dtype=np.int64)
    b = rng.integers(0, Q, size=(3, N), dtype=np.int64)

    assert np.array_equal(poly_add(a, b)[1], poly_add(a[1], b[1]))
    assert np.array_equal(poly_sub(a, b)[2], poly_sub(a[2], b[2]))


@pytest.mark.parametrize("operation", [poly_add, poly_sub, schoolbook_mul])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_operations_do_not_modify_input_and_return_int64(operation, dtype):
    rng = np.random.default_rng(3)
    a, b = random_poly(rng).astype(dtype), random_poly(rng).astype(dtype)
    a_before, b_before = a.copy(), b.copy()

    result = operation(a, b)

    assert np.array_equal(a, a_before)
    assert np.array_equal(b, b_before)
    assert result.dtype == np.int64
    assert result.shape == (N,)
    assert result.min() >= 0
    assert result.max() < Q


def test_schoolbook_wraps_with_sign_change():
    assert np.array_equal(schoolbook_mul(monomial(1), monomial(255)), monomial(0, Q - 1))
    assert np.array_equal(schoolbook_mul(monomial(100), monomial(200)), monomial(44, Q - 1))


def test_schoolbook_without_wrap():
    assert np.array_equal(schoolbook_mul(monomial(3), monomial(4)), monomial(7))


def test_schoolbook_identity_and_zero():
    rng = np.random.default_rng(4)
    a = random_poly(rng)

    assert np.array_equal(schoolbook_mul(a, monomial(0)), a)
    assert np.array_equal(schoolbook_mul(a, np.zeros(N, dtype=np.int64)), np.zeros(N))


def test_schoolbook_is_commutative_and_distributive():
    rng = np.random.default_rng(5)
    a, b, c = random_poly(rng), random_poly(rng), random_poly(rng)

    assert np.array_equal(schoolbook_mul(a, b), schoolbook_mul(b, a))
    assert np.array_equal(
        schoolbook_mul(a, poly_add(b, c)),
        poly_add(schoolbook_mul(a, b), schoolbook_mul(a, c)),
    )


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_schoolbook_largest_coefficients_do_not_overflow(dtype):
    a = np.full(N, Q - 1, dtype=dtype)

    # Q - 1 = -1, so a^2 = (1 + X + ... + X^255)^2, whose k-th wrapped coefficient is 2k - 254.
    expected = (2 * np.arange(N, dtype=np.int64) - 254) % Q
    assert np.array_equal(schoolbook_mul(a, a), expected)


@pytest.mark.parametrize("seed", range(20))
def test_schoolbook_matches_convolve(seed):
    rng = np.random.default_rng(seed)
    a, b = random_poly(rng), random_poly(rng)

    assert np.array_equal(schoolbook_mul(a, b), convolve_mul(a, b))


@pytest.mark.parametrize("seed", range(5))
def test_schoolbook_matches_kyber_py(seed):
    rng = np.random.default_rng(seed)
    a, b = random_poly(rng), random_poly(rng)

    reference = KYBER_RING(a.tolist()) * KYBER_RING(b.tolist())
    assert schoolbook_mul(a, b).tolist() == reference.coeffs


@pytest.mark.parametrize("operation", [poly_add, poly_sub, schoolbook_mul])
@pytest.mark.parametrize(
    ("a", "b"),
    [
        (np.zeros(N), np.zeros(N - 1)),
        (np.zeros(N + 1), np.zeros(N + 1)),
        (np.zeros(N), np.int64(0)),
    ],
)
def test_operations_reject_wrong_shapes(operation, a, b):
    with pytest.raises(ValueError):
        operation(a, b)


def test_schoolbook_rejects_vectors():
    with pytest.raises(ValueError):
        schoolbook_mul(np.zeros((3, N)), np.zeros((3, N)))
