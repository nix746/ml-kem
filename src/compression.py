"""Coefficient compression and byte encoding (FIPS 203, Section 4.2.1)."""

import numpy as np

from params import Q

def _check_compression_d(d: int) -> None:
    """Raises ValueError unless 1 <= d < 12 (the range where Compress_d is defined)."""
    if not 1 <= d < 12:
        raise ValueError(f"d must satisfy 1 <= d < 12, got d={d}")

def compress(d: int, x: np.ndarray) -> np.ndarray:
    """Compresses coefficients from Z_q to Z_{2^d}: round((2^d / q) * x) mod 2^d."""
    _check_compression_d(d)
    x = np.asarray(x, dtype=np.int64)
    if np.any(x < 0) | np.any(x >= Q):
        bad = x[(x < 0) | (x >= Q)]
        raise ValueError(f"compress expects coefficients in [0, Q), got {bad}")
    
    # round(2^d * x / Q) with integers only: add Q // 2 before the floor division
    rounded = ((x << d) + Q // 2) // Q
    return rounded & ((1 << d) - 1)


def decompress(d: int, y: np.ndarray) -> np.ndarray:
    """Decompresses coefficients from Z_{2^d} to Z_q: round((q / 2^d) * y)."""
    _check_compression_d(d)
    y = np.asarray(y, dtype=np.int64)
    if np.any(y < 0) | np.any(y >= (1 << d)):
        bad = y[(y < 0) | (y >= (1 << d))]
        raise ValueError(f"decompress expects coefficients in [0, {1 << d}), got {bad}")
    
    # round(Q * y / 2^d) with integers only: add 2^(d-1) (half of 2^d), then divide by 2^d
    rounded = (Q * y + (1 << (d - 1))) // (1 << d)
    return rounded


def byte_encode(d: int, f: np.ndarray) -> bytes:
    """Encodes an array of 256 d-bit integers into a 32*d-byte array (Algorithm 5)."""
    raise NotImplementedError("TODO: Implement Algorithm 5 (ByteEncode_d) from FIPS 203.")


def byte_decode(d: int, b: bytes) -> np.ndarray:
    """Decodes a 32*d-byte array into an array of 256 integers (Algorithm 6)."""
    raise NotImplementedError("TODO: Implement Algorithm 6 (ByteDecode_d) from FIPS 203.")
