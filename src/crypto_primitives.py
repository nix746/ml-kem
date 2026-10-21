"""Prymitywy kryptograficzne oparte na SHA-3 / Keccak (NIST FIPS 203, Sekcja 4.1)."""


def h_hash(data: bytes) -> bytes:
    """Funkcja skrótu H(s) = SHA3-256(s)."""
    raise NotImplementedError("TODO: Implementacja H (SHA3-256) z FIPS 203 (Sekcja 4.1).")


def j_hash(data: bytes) -> bytes:
    """Funkcja J(s) = SHAKE256(s, 32)."""
    raise NotImplementedError("TODO: Implementacja J (SHAKE256) z FIPS 203 (Sekcja 4.1).")


def g_hash(data: bytes) -> tuple[bytes, bytes]:
    """Funkcja skrótu G(c) = SHA3-512(c)."""
    raise NotImplementedError("TODO: Implementacja G (SHA3-512) z FIPS 203 (Sekcja 4.1).")


def prf(eta: int, s: bytes, b: int) -> bytes:
    """Funkcja pseudolosowa PRF_eta(s, b) = SHAKE256(s || b, 64 * eta)."""
    raise NotImplementedError("TODO: Implementacja PRF z FIPS 203 (Sekcja 4.1).")


def xof(rho: bytes, i: int, j: int):
    """Funkcja rozszerzalnego wyjścia XOF(rho, i, j) = SHAKE128(rho || i || j)."""
    raise NotImplementedError("TODO: Implementacja XOF z FIPS 203 (Sekcja 4.1).")
