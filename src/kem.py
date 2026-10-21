"""Full ML-KEM scheme (IND-CCA2) with the Fujisaki-Okamoto transform (FIPS 203, Sec. 6-7)."""

from params import MLKEMParams


def ml_kem_keygen(params: MLKEMParams) -> tuple[bytes, bytes]:
    """Generates the ML-KEM key pair (ek, dk) (Algorithms 16 and 19)."""
    raise NotImplementedError("TODO: Implement Algorithm 16 / 19 (ML-KEM.KeyGen) from FIPS 203.")


def ml_kem_encaps(params: MLKEMParams, ek: bytes) -> tuple[bytes, bytes]:
    """Encapsulates a shared secret K and returns the pair (K, c) (Algorithms 17 and 20)."""
    raise NotImplementedError("TODO: Implement Algorithm 17 / 20 (ML-KEM.Encaps) from FIPS 203.")


def ml_kem_decaps(params: MLKEMParams, dk: bytes, c: bytes) -> bytes:
    """Decapsulates ciphertext c (with FO re-encryption check) and returns K (Alg. 18 and 21)."""
    raise NotImplementedError("TODO: Implement Algorithm 18 / 21 (ML-KEM.Decaps) from FIPS 203.")
