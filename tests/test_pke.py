"""Tests for the internal K-PKE encryption scheme (FIPS 203, Section 5)."""

import pytest
from kyber_py.ml_kem.default_parameters import DEFAULT_PARAMETERS
from kyber_py.ml_kem.ml_kem import ML_KEM

from params import ML_KEM_512, ML_KEM_768, ML_KEM_1024, MLKEMParams
from pke import k_pke_decrypt, k_pke_encrypt, k_pke_keygen


def _kyber_instance(params: MLKEMParams) -> ML_KEM:
    return ML_KEM(DEFAULT_PARAMETERS[params.name.replace("ML-KEM-", "ML")])


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_k_pke_matches_kyber_py_oracle(params):
    """Verifies that k_pke_* matches kyber-py bit-for-bit on fixed seeds."""
    d = bytes(range(32))
    r = bytes(range(32, 64))
    m = bytes(range(64, 96))

    oracle = _kyber_instance(params)

    # KeyGen bit-for-bit check
    ek_pke, dk_pke = k_pke_keygen(params, d)
    ek_exp, dk_exp = oracle._k_pke_keygen(d)
    assert ek_pke == ek_exp
    assert dk_pke == dk_exp

    # Encrypt bit-for-bit check
    c = k_pke_encrypt(params, ek_pke, m, r)
    c_exp = oracle._k_pke_encrypt(ek_exp, m, r)
    assert c == c_exp

    # Decrypt check
    m_recovered = k_pke_decrypt(params, dk_pke, c)
    m_exp = oracle._k_pke_decrypt(dk_exp, c_exp)
    assert m_recovered == m_exp
    assert m_recovered == m


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_k_pke_encrypt_decrypt_roundtrip(params):
    """Verifies Decrypt(Encrypt(m)) == m across all three ML-KEM parameter sets."""
    d = bytes(range(32))
    r = bytes(range(32, 64))
    m = bytes(range(64, 96))

    ek_pke, dk_pke = k_pke_keygen(params, d)
    c = k_pke_encrypt(params, ek_pke, m, r)
    m_recovered = k_pke_decrypt(params, dk_pke, c)

    assert len(ek_pke) == params.ek_size
    assert len(dk_pke) == 384 * params.k
    assert len(c) == params.ct_size
    assert m_recovered == m


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_k_pke_deterministic(params):
    """Checks that identical seeds always produce identical keys and ciphertexts."""
    d = b"\x01" * 32
    r = b"\x02" * 32
    m = b"\x03" * 32

    ek1, dk1 = k_pke_keygen(params, d)
    ek2, dk2 = k_pke_keygen(params, d)
    assert ek1 == ek2
    assert dk1 == dk2

    c1 = k_pke_encrypt(params, ek1, m, r)
    c2 = k_pke_encrypt(params, ek2, m, r)
    assert c1 == c2
