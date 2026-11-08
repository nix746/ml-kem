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


def _check_byte_encode_d(d: int) -> None:
    """Raises ValueError unless 1 <= d < 12 (the range where ByteEncode_d is defined)."""
    if not 1 <= d < 12:
        raise ValueError(f"d must satisfy 1 <= d < 12, got d={d}")

def _encoding_modulus(d: int) -> int:
    """The modulus m of ByteEncode_d / ByteDecode_d: 2^d for d < 12 and Q for d = 12."""
    return Q if d == 12 else 1 << d

# Weights of the 8 bit positions inside one byte: 2^0, 2^1, ..., 2^7 (little-endian bit order).
_BYTE_BIT_WEIGHTS = 1 << np.arange(8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Converts a bit array of length 8*l into l bytes (FIPS 203, Algorithm 3).

    Bit i goes to byte i // 8 with weight 2^(i mod 8): the first bit is the least
    significant bit of the first byte.
    """
    bits = np.asarray(bits, dtype=np.int64)
    if bits.ndim != 1 or bits.size % 8 != 0:
        raise ValueError("bits_to_bytes expects a 1-D bit array whose length is a multiple of 8")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits_to_bytes expects only the values 0 and 1")

    # Row r holds the 8 bits of byte r; byte = sum of bit_j * 2^j over its row.
    byte_values = (bits.reshape(-1, 8) * _BYTE_BIT_WEIGHTS).sum(axis=1)
    return byte_values.astype(np.uint8).tobytes()

def bytes_to_bits(b: bytes) -> np.ndarray:
    """Converts a bytes object of length l into a bit array of length 8*l (FIPS 203, Algorithm 4).

    Bit i comes from byte i // 8 with weight 2^(i mod 8): the first bit is the least
    significant bit of the first byte.
    """
    byte_values = np.frombuffer(b, dtype=np.uint8)
    bits = ((byte_values[:, None] & _BYTE_BIT_WEIGHTS) > 0).astype(np.int64)
    return bits.reshape(-1)  # flatten to a 1-D array

def byte_encode(d: int, f: np.ndarray) -> bytes:
    """Encodes an array of 256 d-bit integers into a 32*d-byte array (Algorithm 5)."""
    _check_byte_encode_d(d)
    f = np.asarray(f, dtype=np.int64)
    if f.shape != (256,):
        raise ValueError(f"byte_encode expects an array of shape (256,), got {f.shape}")
    if np.any(f < 0) | np.any(f >= _encoding_modulus(d)):
        bad = f[(f < 0) | (f >= _encoding_modulus(d))]
        raise ValueError(f"byte_encode expects coefficients in [0, {_encoding_modulus(d)}), got {bad}")

    bits = (f[:, None] >> np.arange(d)) & 1  # shape (256, d)
    return bits_to_bytes(bits.reshape(-1))  # flatten to (256*d,) and convert to bytes

def byte_decode(d: int, b: bytes) -> np.ndarray:
    """Decodes a 32*d-byte array into an array of 256 integers (Algorithm 6)."""
    _check_byte_encode_d(d)
    if len(b) != 32 * d:
        raise ValueError(f"byte_decode expects a bytes object of length {32*d}, got {len(b)}")
    
    bits = bytes_to_bits(b).reshape(256, d)  # shape (256, d)
    f = (bits * (1 << np.arange(d))).sum(axis=1)  # sum over d to get integers
    return f % _encoding_modulus(d)  # ensure values are in the correct range
