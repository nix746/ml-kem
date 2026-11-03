"""Tests for the SHA-3 based primitives against the kyber-py reference implementation."""

import pytest
from kyber_py.ml_kem import ML_KEM_768 as KYBER_ML_KEM_768
from kyber_py.ml_kem.default_parameters import DEFAULT_PARAMETERS
from kyber_py.ml_kem.ml_kem import ML_KEM

from crypto_primitives import g_hash, h_hash, j_hash, prf, xof
from params import ML_KEM_512, ML_KEM_768, ML_KEM_1024

SEED = bytes(range(32))


@pytest.mark.parametrize("data", [b"", b"abc", SEED, bytes(1000)])
def test_hashes_match_kyber_py(data):
    assert h_hash(data) == KYBER_ML_KEM_768._H(data)
    assert j_hash(data) == KYBER_ML_KEM_768._J(data)
    assert g_hash(data) == KYBER_ML_KEM_768._G(data)


def test_hash_output_lengths():
    assert len(h_hash(SEED)) == 32
    assert len(j_hash(SEED)) == 32
    first, second = g_hash(SEED)
    assert (len(first), len(second)) == (32, 32)


@pytest.mark.parametrize("eta", [2, 3])
@pytest.mark.parametrize("b", [0, 1, 255])
def test_prf_matches_kyber_py(eta, b):
    output = prf(eta, SEED, b)
    assert output == KYBER_ML_KEM_768._prf(eta, SEED, bytes([b]))
    assert len(output) == 64 * eta


def test_xof_matches_kyber_py():
    i, j = 1, 2
    assert xof(SEED + bytes([i, j]), 840) == KYBER_ML_KEM_768._xof(SEED, bytes([i]), bytes([j]))


def test_xof_longer_output_extends_shorter_one():
    assert xof(SEED, 840)[:168] == xof(SEED, 168)
    assert xof(SEED, 0) == b""


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_g_with_k_domain_separation_gives_rho_of_ek(params):
    kyber_name = params.name.replace("ML-KEM-", "ML")
    kem = ML_KEM(DEFAULT_PARAMETERS[kyber_name])
    ek_pke, _ = kem._k_pke_keygen(SEED)

    rho, _ = g_hash(SEED + bytes([params.k]))
    assert rho == ek_pke[-32:]
    assert g_hash(SEED)[0] != ek_pke[-32:]


@pytest.mark.parametrize(
    ("eta", "s", "b"),
    [
        (1, SEED, 0),
        (4, SEED, 0),
        (2, SEED[:31], 0),
        (2, SEED + b"\x00", 0),
        (2, SEED, -1),
        (2, SEED, 256),
    ],
)
def test_prf_rejects_invalid_arguments(eta, s, b):
    with pytest.raises(ValueError):
        prf(eta, s, b)


def test_xof_rejects_negative_length():
    with pytest.raises(ValueError):
        xof(SEED, -1)
