"""Testy weryfikujące poprawne działanie bibliotek środowiska (numpy, hashlib, kyber-py)."""

import hashlib

import numpy as np
from kyber_py.ml_kem import ML_KEM_768


def test_numpy_environment():
    """Sprawdza działanie biblioteki NumPy na operacjach modulo."""
    a = np.array([3328, 10, 500], dtype=np.int32)
    b = np.array([2, 20, 3000], dtype=np.int32)
    res = (a + b) % 3329
    assert np.array_equal(res, np.array([1, 30, 171], dtype=np.int32))


def test_hashlib_sha3_and_shake_environment():
    """Sprawdza dostępność SHA3 i SHAKE w standardowej bibliotece hashlib."""
    seed = b"test_seed_32_bytes______________"

    assert len(hashlib.sha3_256(seed).digest()) == 32
    assert len(hashlib.sha3_512(seed).digest()) == 64
    assert len(hashlib.shake_128(seed).digest(168)) == 168
    assert len(hashlib.shake_256(seed).digest(128)) == 128


def test_reference_oracle_kyber_py():
    """Sprawdza działanie biblioteki wzorcowej kyber-py (wyroczni testowej)."""
    ek, dk = ML_KEM_768.keygen()
    k_alice, c = ML_KEM_768.encaps(ek)
    k_bob = ML_KEM_768.decaps(dk, c)

    assert k_alice == k_bob
    assert len(k_alice) == 32
