"""Cryptographic primitives based on SHA-3 / Keccak (NIST FIPS 203, Section 4.1)."""


def h_hash(data: bytes) -> bytes:
    """Hash function H(s) = SHA3-256(s)."""
    raise NotImplementedError("TODO: Implement H (SHA3-256) from FIPS 203 (Section 4.1).")


def j_hash(data: bytes) -> bytes:
    """Function J(s) = SHAKE256(s, 32)."""
    raise NotImplementedError("TODO: Implement J (SHAKE256) from FIPS 203 (Section 4.1).")


def g_hash(data: bytes) -> tuple[bytes, bytes]:
    """Hash function G(c) = SHA3-512(c)."""
    raise NotImplementedError("TODO: Implement G (SHA3-512) from FIPS 203 (Section 4.1).")


def prf(eta: int, s: bytes, b: int) -> bytes:
    """Pseudorandom function PRF_eta(s, b) = SHAKE256(s || b, 64 * eta)."""
    raise NotImplementedError("TODO: Implement PRF from FIPS 203 (Section 4.1).")


def xof(rho: bytes, i: int, j: int):
    """Extendable-output function XOF(rho, i, j) = SHAKE128(rho || i || j)."""
    raise NotImplementedError("TODO: Implement XOF from FIPS 203 (Section 4.1).")
