"""Tests for the full ML-KEM mechanism (FIPS 203, Sections 6 and 7)."""

import pytest
from kyber_py.ml_kem.default_parameters import DEFAULT_PARAMETERS
from kyber_py.ml_kem.ml_kem import ML_KEM

import kem
from crypto_primitives import j_hash
from kem import (
    ml_kem_decaps,
    ml_kem_decaps_internal,
    ml_kem_encaps,
    ml_kem_encaps_internal,
    ml_kem_keygen,
    ml_kem_keygen_internal,
)
from params import ML_KEM_512, ML_KEM_768, ML_KEM_1024, MLKEMParams


def _kyber_instance(params: MLKEMParams) -> ML_KEM:
    return ML_KEM(DEFAULT_PARAMETERS[params.name.replace("ML-KEM-", "ML")])


@pytest.fixture(autouse=True)
def _fallback_unimplemented_pke(monkeypatch: pytest.MonkeyPatch) -> None:
    # Temporary fallback to kyber-py until k-pke and sampling/compression branches are merged.
    try:
        kem.k_pke_keygen(ML_KEM_512, bytes(32))
    except NotImplementedError:
        monkeypatch.setattr(
            kem,
            "k_pke_keygen",
            lambda params, d: _kyber_instance(params)._k_pke_keygen(d),
        )
        monkeypatch.setattr(
            kem,
            "k_pke_encrypt",
            lambda params, ek_pke, m, r: _kyber_instance(params)._k_pke_encrypt(ek_pke, m, r),
        )
        monkeypatch.setattr(
            kem,
            "k_pke_decrypt",
            lambda params, dk_pke, c: _kyber_instance(params)._k_pke_decrypt(dk_pke, c),
        )


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_ml_kem_roundtrip_and_implicit_rejection(params):
    ek, dk = ml_kem_keygen(params)
    k_enc, c = ml_kem_encaps(params, ek)
    k_dec = ml_kem_decaps(params, dk, c)

    assert len(ek) == params.ek_size
    assert len(dk) == params.dk_size
    assert len(c) == params.ct_size
    assert len(k_enc) == 32
    assert k_dec == k_enc

    c_tampered = bytes([c[0] ^ 1]) + c[1:]
    k_rejected = ml_kem_decaps(params, dk, c_tampered)
    assert k_rejected == j_hash(dk[-32:] + c_tampered)
    assert k_rejected != k_enc


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_ml_kem_internal_deterministic_matches_oracle(params):
    d = bytes(range(32))
    z = bytes(range(32, 64))
    m = bytes(range(64, 96))

    ek, dk = ml_kem_keygen_internal(params, d, z)
    k_shared, c = ml_kem_encaps_internal(params, ek, m)
    k_dec = ml_kem_decaps_internal(params, dk, c)

    oracle = _kyber_instance(params)
    ek_exp, dk_exp = oracle._keygen_internal(d, z)
    k_exp, c_exp = oracle._encaps_internal(ek_exp, m)

    assert ek == ek_exp
    assert dk == dk_exp
    assert c == c_exp
    assert k_shared == k_exp
    assert k_dec == k_exp
