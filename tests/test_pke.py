"""Tests for the internal K-PKE encryption scheme (FIPS 203, Section 5)."""

import numpy as np
import pytest
from kyber_py.polynomials.polynomials import PolynomialRing

import pke
from crypto_primitives import xof
from params import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from pke import k_pke_decrypt, k_pke_encrypt, k_pke_keygen

KYBER_RING = PolynomialRing()


@pytest.fixture(autouse=True)
def _fallback_unimplemented_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bridges sampling/compression via kyber-py until Person 2's branch is merged."""
    try:
        pke.sample_ntt(bytes(32), 0, 0)
    except NotImplementedError:
        monkeypatch.setattr(
            pke,
            "sample_ntt",
            lambda rho, i, j: np.array(
                KYBER_RING.ntt_sample(xof(rho + bytes([j, i]), 840)).coeffs,
                dtype=np.int64,
            ),
        )
        monkeypatch.setattr(
            pke,
            "sample_poly_cbd",
            lambda eta, buf: np.array(KYBER_RING.cbd(buf, eta).coeffs, dtype=np.int64),
        )

    try:
        pke.compress(1, np.zeros(256, dtype=np.int64))
    except NotImplementedError:
        monkeypatch.setattr(
            pke,
            "compress",
            lambda d, x: np.array(
                KYBER_RING(np.asarray(x, dtype=int).tolist()).compress(d).coeffs,
                dtype=np.int64,
            ),
        )
        monkeypatch.setattr(
            pke,
            "decompress",
            lambda d, y: np.array(
                KYBER_RING(np.asarray(y, dtype=int).tolist()).decompress(d).coeffs,
                dtype=np.int64,
            ),
        )
        monkeypatch.setattr(
            pke,
            "byte_encode",
            lambda d, f: KYBER_RING(np.asarray(f, dtype=int).tolist()).encode(d),
        )
        monkeypatch.setattr(
            pke,
            "byte_decode",
            lambda d, b: np.array(KYBER_RING.decode(b, d).coeffs, dtype=np.int64),
        )


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

