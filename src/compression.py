"""Kompresja współczynników i kodowanie bajtowe (FIPS 203, Sekcja 4.2.1)."""

import numpy as np


def compress(d: int, x: np.ndarray) -> np.ndarray:
    """Kompresuje współczynniki z Z_q do Z_{2^d}: Compress_d(x) = round((2^d / q) * x) mod 2^d."""
    raise NotImplementedError("TODO: Implementacja funkcji Compress_d z FIPS 203 (Sekcja 4.2.1).")


def decompress(d: int, y: np.ndarray) -> np.ndarray:
    """Dekompresuje współczynniki z Z_{2^d} do Z_q: Decompress_d(y) = round((q / 2^d) * y)."""
    raise NotImplementedError("TODO: Implementacja funkcji Decompress_d z FIPS 203 (Sekcja 4.2.1).")


def byte_encode(d: int, f: np.ndarray) -> bytes:
    """Koduje tablicę 256 d-bitowych liczb całkowitych do tablicy 32*d bajtów (Algorithm 5)."""
    raise NotImplementedError("TODO: Implementacja Algorithm 5 (ByteEncode_d) z FIPS 203.")


def byte_decode(d: int, b: bytes) -> np.ndarray:
    """Dekoduje tablicę 32*d bajtów do tablicy 256 liczb całkowitych (Algorithm 6)."""
    raise NotImplementedError("TODO: Implementacja Algorithm 6 (ByteDecode_d) z FIPS 203.")
