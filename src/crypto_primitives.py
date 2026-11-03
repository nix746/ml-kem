"""Cryptographic primitives based on SHA-3 / Keccak (NIST FIPS 203, Section 4.1)."""

import hashlib


def h_hash(data: bytes) -> bytes:
    """Hash function H(s) = SHA3-256(s), 32 bytes."""
    return hashlib.sha3_256(data).digest()


def j_hash(data: bytes) -> bytes:
    """Function J(s) = SHAKE256(s, 32), 32 bytes."""
    return hashlib.shake_256(data).digest(32)


def g_hash(data: bytes) -> tuple[bytes, bytes]:
    """Hash function G(c) = SHA3-512(c), split into two 32-byte halves.

    KeyGen must call this with `d + bytes([k])`, not `d` alone (FIPS 203, Algorithm 13, line 1):
    the first half is then rho, which equals the last 32 bytes of `ek_pke`.
    """
    digest = hashlib.sha3_512(data).digest()
    return digest[:32], digest[32:]


def prf(eta: int, s: bytes, b: int) -> bytes:
    """Pseudorandom function PRF_eta(s, b) = SHAKE256(s || b, 64 * eta)."""
    if eta not in {2, 3}:
        raise ValueError(f"eta must be 2 or 3, got {eta}")
    if len(s) != 32:
        raise ValueError(f"s must be exactly 32 bytes, got {len(s)}")
    if not 0 <= b <= 255:
        raise ValueError(f"b must be in [0, 255], got {b}")
    return hashlib.shake_256(s + bytes([b])).digest(64 * eta)


def xof(seed: bytes, length: int) -> bytes:
    """Extendable-output function XOF(seed) = SHAKE128(seed), truncated to `length` bytes.

    The caller builds the seed: `sample_ntt` passes `rho + bytes([j, i])` (CONTRACT, Section 5.1).
    A longer output starts with the shorter one, so asking for more bytes keeps the prefix.
    """
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    return hashlib.shake_128(seed).digest(length)
