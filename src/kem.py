"""Full ML-KEM scheme (IND-CCA2) with the Fujisaki-Okamoto transform (FIPS 203, Sec. 6-7)."""

import secrets

from crypto_primitives import g_hash, h_hash, j_hash
from params import MLKEMParams
from pke import k_pke_decrypt, k_pke_encrypt, k_pke_keygen


def ml_kem_keygen(params: MLKEMParams) -> tuple[bytes, bytes]:
    """Generates the ML-KEM key pair (ek, dk) (Algorithms 16 and 19)."""
    d = secrets.token_bytes(32)
    z = secrets.token_bytes(32)

    ek_pke, dk_pke = k_pke_keygen(params, d)
    ek = ek_pke
    dk = dk_pke + ek + h_hash(ek) + z
    return ek, dk

def ml_kem_encaps(params: MLKEMParams, ek: bytes) -> tuple[bytes, bytes]:
    """Encapsulates a shared secret K and returns the pair (K, c) (Algorithms 17 and 20)."""
    # Prototype simplification: FIPS 203 Section 7.2 encapsulation key validation is omitted.
    m = secrets.token_bytes(32)
    k_shared, r = g_hash(m + h_hash(ek))
    c = k_pke_encrypt(params, ek, m, r)
    return k_shared, c


def ml_kem_decaps(params: MLKEMParams, dk: bytes, c: bytes) -> bytes:
    """Decapsulates ciphertext c (with FO re-encryption check) and returns K (Alg. 18 and 21)."""
    # Prototype simplification: FIPS 203 Section 7.3 decapsulation input validation is omitted.
    k = params.k
    dk_pke = dk[: 384 * k]
    ek_pke = dk[384 * k : 768 * k + 32]
    h = dk[768 * k + 32 : 768 * k + 64]
    z = dk[768 * k + 64 : 768 * k + 96]

    m_prime = k_pke_decrypt(params, dk_pke, c)
    k_prime, r_prime = g_hash(m_prime + h)
    k_bar = j_hash(z + c)
    c_prime = k_pke_encrypt(params, ek_pke, m_prime, r_prime)

    return k_prime if secrets.compare_digest(c, c_prime) else k_bar
