"""Coefficient compression and byte encoding (FIPS 203, Section 4.2.1)."""

import numpy as np


def compress(d: int, x: np.ndarray) -> np.ndarray:
    """Compresses coefficients from Z_q to Z_{2^d}: round((2^d / q) * x) mod 2^d."""
    raise NotImplementedError("TODO: Implement Compress_d from FIPS 203 (Section 4.2.1).")


def decompress(d: int, y: np.ndarray) -> np.ndarray:
    """Decompresses coefficients from Z_{2^d} to Z_q: round((q / 2^d) * y)."""
    raise NotImplementedError("TODO: Implement Decompress_d from FIPS 203 (Section 4.2.1).")


def byte_encode(d: int, f: np.ndarray) -> bytes:
    """Encodes an array of 256 d-bit integers into a 32*d-byte array (Algorithm 5)."""
    raise NotImplementedError("TODO: Implement Algorithm 5 (ByteEncode_d) from FIPS 203.")


def byte_decode(d: int, b: bytes) -> np.ndarray:
    """Decodes a 32*d-byte array into an array of 256 integers (Algorithm 6)."""
    raise NotImplementedError("TODO: Implement Algorithm 6 (ByteDecode_d) from FIPS 203.")
