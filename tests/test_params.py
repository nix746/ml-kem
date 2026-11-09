"""Tests for ML-KEM parameters and their reference implementation."""

from kyber_py.ml_kem import ML_KEM_512 as KYBER_ML_KEM_512
from kyber_py.ml_kem import ML_KEM_768 as KYBER_ML_KEM_768
from kyber_py.ml_kem import ML_KEM_1024 as KYBER_ML_KEM_1024
from kyber_py.ml_kem.default_parameters import DEFAULT_PARAMETERS

from params import (
    ML_KEM_512,
    ML_KEM_768,
    ML_KEM_1024,
    N_INV,
    ZETA,
    MLKEMParams,
    N,
    Q,
)


def test_ring_constants():
    assert pow(ZETA, N, Q) == 1
    assert pow(ZETA, N // 2, Q) == Q - 1
    assert N_INV == 3303
    assert 128 * N_INV % Q == 1


def test_parameter_sets_match_kyber_py_and_encoded_sizes():
    parameter_sets = (
        (ML_KEM_512, "ML512", KYBER_ML_KEM_512),
        (ML_KEM_768, "ML768", KYBER_ML_KEM_768),
        (ML_KEM_1024, "ML1024", KYBER_ML_KEM_1024),
    )

    for params, kyber_name, kyber_kem in parameter_sets:
        assert isinstance(params, MLKEMParams)
        reference = DEFAULT_PARAMETERS[kyber_name]
        assert (params.k, params.eta1, params.eta2, params.du, params.dv) == (
            reference["k"],
            reference["eta_1"],
            reference["eta_2"],
            reference["du"],
            reference["dv"],
        )

        ek, dk = kyber_kem.keygen()
        _, ciphertext = kyber_kem.encaps(ek)
        assert params.ek_size == len(ek)
        assert params.dk_size == len(dk)
        assert params.ct_size == len(ciphertext)
