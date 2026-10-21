"""Pełny schemat ML-KEM (IND-CCA2) z transformacją Fujisaki-Okamoto (FIPS 203, Sekcja 6 i 7)."""

from params import MLKEMParams


def ml_kem_keygen(params: MLKEMParams) -> tuple[bytes, bytes]:
    """Generuje parę kluczy (ek, dk) dla ML-KEM (Algorithm 16 i 19)."""
    raise NotImplementedError("TODO: Implementacja Algorithm 16 / 19 (ML-KEM.KeyGen) z FIPS 203.")


def ml_kem_encaps(params: MLKEMParams, ek: bytes) -> tuple[bytes, bytes]:
    """Enkapsuluje wspólny sekret K i zwraca parę (K, c) (Algorithm 17 i 20)."""
    raise NotImplementedError("TODO: Implementacja Algorithm 17 / 20 (ML-KEM.Encaps) z FIPS 203.")


def ml_kem_decaps(params: MLKEMParams, dk: bytes, c: bytes) -> bytes:
    """Dekapsuluje szyfrogram c (z weryfikacją FO) i zwraca K (Algorithm 18 i 21)."""
    raise NotImplementedError("TODO: Implementacja Algorithm 18 / 21 (ML-KEM.Decaps) z FIPS 203.")
