"""Tests for Compress/Decompress and ByteEncode/ByteDecode (FIPS 203, Section 4.2.1)."""

import numpy as np
import pytest
from kyber_py.ml_kem import ML_KEM_768

from compression import (
    bits_to_bytes,
    byte_decode,
    byte_encode,
    bytes_to_bits,
    compress,
    decompress,
)
from params import Q

# The values of d that actually occur in ML-KEM: 1 (message), du and dv of all three sets.
D_VALUES = [1, 4, 5, 10, 11]

# Any kyber-py polynomial gives access to its per-coefficient compression helpers.
REF = ML_KEM_768.R.zero()

ALL_X = np.arange(Q)  # every element of Z_q: 0, 1, ..., 3328


# --- Comparison with the reference oracle (kyber-py) -------------------------------------


@pytest.mark.parametrize("d", D_VALUES)
def test_compress_matches_kyber_py_for_every_x(d):
    expected = [REF._compress_ele(x, d) for x in range(Q)]
    assert compress(d, ALL_X).tolist() == expected


@pytest.mark.parametrize("d", D_VALUES)
def test_decompress_matches_kyber_py_for_every_y(d):
    all_y = np.arange(1 << d)
    expected = [REF._decompress_ele(y, d) for y in range(1 << d)]
    assert decompress(d, all_y).tolist() == expected


# --- Mathematical properties ----------------------------------------------------------------


def centered_distance(a, b):
    """Distance between a and b on the circle Z_q, i.e. |a - b| mod+- q."""
    diff = (a - b) % Q
    return np.minimum(diff, Q - diff)


# round(Q / 2^(d+1)): the largest error that Decompress(Compress(x)) can introduce.
ERROR_BOUND = {1: 832, 4: 104, 5: 52, 10: 2, 11: 1}


@pytest.mark.parametrize("d", D_VALUES)
def test_compression_error_is_bounded(d):
    x_back = decompress(d, compress(d, ALL_X))
    error = centered_distance(ALL_X, x_back)
    assert error.max() <= ERROR_BOUND[d]


@pytest.mark.parametrize("d", D_VALUES)
def test_compress_after_decompress_is_identity(d):
    # The other direction is exact: no information is lost when going 2^d -> q -> 2^d.
    all_y = np.arange(1 << d)
    assert np.array_equal(compress(d, decompress(d, all_y)), all_y)


@pytest.mark.parametrize("d", D_VALUES)
def test_output_ranges(d):
    assert compress(d, ALL_X).min() >= 0
    assert compress(d, ALL_X).max() < (1 << d)
    all_y = np.arange(1 << d)
    assert decompress(d, all_y).min() >= 0
    assert decompress(d, all_y).max() < Q


def test_d1_is_the_message_decoder():
    # Compress_1 returns 1 exactly when x is closer to Q/2 than to 0 (on the circle).
    expected = ((ALL_X >= 833) & (ALL_X <= 2496)).astype(np.int64)
    assert np.array_equal(compress(1, ALL_X), expected)
    # Decompress_1 maps bit 0 -> 0 and bit 1 -> 1665 (about Q/2).
    assert decompress(1, np.array([0, 1])).tolist() == [0, 1665]


def test_examples_by_hand():
    assert compress(4, np.array([1000])).tolist() == [5]
    assert decompress(4, np.array([5])).tolist() == [1040]
    # 3328 is "almost -1", i.e. almost 0 on the circle, so it wraps around to 0.
    assert compress(10, np.array([3328])).tolist() == [0]


# --- Shapes and input validation ------------------------------------------------------------


def test_works_on_vectors_of_polynomials():
    rng = np.random.default_rng(2026)
    u = rng.integers(0, Q, size=(3, 256))  # a vector in R_q^3, like u in K-PKE.Encrypt
    c = compress(10, u)
    assert c.shape == (3, 256)
    assert decompress(10, c).shape == (3, 256)


@pytest.mark.parametrize("bad_d", [0, 12, -1])
def test_rejects_invalid_d(bad_d):
    with pytest.raises(ValueError):
        compress(bad_d, np.array([0]))
    with pytest.raises(ValueError):
        decompress(bad_d, np.array([0]))


def test_rejects_values_out_of_range():
    with pytest.raises(ValueError):
        compress(4, np.array([Q]))
    with pytest.raises(ValueError):
        compress(4, np.array([-1]))
    with pytest.raises(ValueError):
        decompress(4, np.array([16]))


# =============================================================================================
# BitsToBytes / BytesToBits (FIPS 203, Algorithms 3 and 4)
# =============================================================================================


def test_bytes_to_bits_example_by_hand():
    # 197 = 128 + 64 + 4 + 1 = 0b11000101, read from the least significant bit.
    assert bytes_to_bits(bytes([197])).tolist() == [1, 0, 1, 0, 0, 0, 1, 1]
    # Second byte starts at bit 8: bytes [1, 128] -> bit 0 and bit 15 are set.
    bits = bytes_to_bits(bytes([1, 128]))
    assert np.flatnonzero(bits).tolist() == [0, 15]


def test_bits_to_bytes_example_by_hand():
    assert bits_to_bytes(np.array([1, 0, 1, 0, 0, 0, 1, 1])) == bytes([197])


def test_bits_bytes_round_trip():
    rng = np.random.default_rng(3)
    b = rng.bytes(96)
    assert bits_to_bytes(bytes_to_bits(b)) == b
    bits = rng.integers(0, 2, size=8 * 40)
    assert np.array_equal(bytes_to_bits(bits_to_bytes(bits)), bits)


def test_bits_to_bytes_rejects_bad_input():
    with pytest.raises(ValueError):
        bits_to_bytes(np.array([1, 0, 1]))  # length not a multiple of 8
    with pytest.raises(ValueError):
        bits_to_bytes(np.array([0, 0, 0, 0, 0, 0, 0, 2]))  # 2 is not a bit


# =============================================================================================
# ByteEncode_d / ByteDecode_d (FIPS 203, Algorithms 5 and 6)
# =============================================================================================

ENCODE_D_VALUES = list(range(1, 13))  # 1, 2, ..., 12


def modulus(d):
    return Q if d == 12 else 1 << d


def random_poly(d, seed):
    """A random array of 256 valid inputs for ByteEncode_d."""
    return np.random.default_rng(seed).integers(0, modulus(d), size=256)


@pytest.mark.parametrize("d", ENCODE_D_VALUES)
def test_byte_encode_matches_kyber_py(d):
    for seed in range(20):
        f = random_poly(d, seed)
        expected = ML_KEM_768.R(f.tolist()).encode(d)
        assert byte_encode(d, f) == expected


@pytest.mark.parametrize("d", ENCODE_D_VALUES)
def test_byte_decode_matches_kyber_py(d):
    rng = np.random.default_rng(d)
    for _ in range(20):
        b = rng.bytes(32 * d)
        expected = ML_KEM_768.R.decode(b, d).coeffs
        assert byte_decode(d, b).tolist() == expected


@pytest.mark.parametrize("d", ENCODE_D_VALUES)
def test_output_lengths(d):
    f = random_poly(d, 0)
    assert len(byte_encode(d, f)) == 32 * d
    assert byte_decode(d, byte_encode(d, f)).shape == (256,)


@pytest.mark.parametrize("d", ENCODE_D_VALUES)
def test_decode_after_encode_is_identity(d):
    # FIPS 203: ByteDecode_d(ByteEncode_d(F)) = F for every valid F.
    for seed in range(20):
        f = random_poly(d, seed)
        assert np.array_equal(byte_decode(d, byte_encode(d, f)), f)


@pytest.mark.parametrize("d", range(1, 12))
def test_encode_after_decode_is_identity_for_d_below_12(d):
    # For d < 12 every byte string is a valid encoding, so the other direction is exact too.
    rng = np.random.default_rng(100 + d)
    for _ in range(20):
        b = rng.bytes(32 * d)
        assert byte_encode(d, byte_decode(d, b)) == b


def test_encode_example_by_hand_d4():
    # d = 4: each byte holds two coefficients, byte = f_0 + 16 * f_1.
    f = np.zeros(256, dtype=np.int64)
    f[0], f[1] = 5, 12
    assert byte_encode(4, f)[0] == 5 + 16 * 12  # = 197


def test_encode_example_by_hand_d12():
    # d = 12: two coefficients a, b share 3 bytes (example from docs/Math-Code.md, 7.1).
    f = np.zeros(256, dtype=np.int64)
    f[0], f[1] = 0x123, 0xABC
    assert byte_encode(12, f)[:3] == bytes([0x23, 0xC1, 0xAB])


def test_decode_d12_reduces_mod_q():
    # The 12-bit value 0xFFF = 4095 is not in Z_q; ByteDecode_12 must return 4095 - Q = 766.
    b = bytearray(32 * 12)
    b[0], b[1] = 0xFF, 0x0F  # f_0 = 0xFFF, all other coefficients 0
    f = byte_decode(12, bytes(b))
    assert f[0] == 4095 - Q == 766
    # This is why the ek "modulus check" works: re-encoding does not give the same bytes.
    assert byte_encode(12, f) != bytes(b)


def test_compress_then_encode_matches_kyber_py():
    # The real pipeline of K-PKE.Encrypt: c = ByteEncode_d(Compress_d(u)), here for du = 10.
    x = np.random.default_rng(7).integers(0, Q, size=256)
    expected = ML_KEM_768.R(x.tolist()).compress(10).encode(10)
    assert byte_encode(10, compress(10, x)) == expected


@pytest.mark.parametrize("bad_d", [0, 13])
def test_encoding_rejects_invalid_d(bad_d):
    with pytest.raises(ValueError):
        byte_encode(bad_d, np.zeros(256, dtype=np.int64))
    with pytest.raises(ValueError):
        byte_decode(bad_d, bytes(32))


def test_encoding_rejects_bad_input():
    with pytest.raises(ValueError):
        byte_encode(4, np.full(256, 16))  # 16 does not fit in 4 bits
    with pytest.raises(ValueError):
        byte_encode(12, np.full(256, Q))  # Q is not in Z_q
    with pytest.raises(ValueError):
        byte_encode(4, np.zeros(255, dtype=np.int64))  # wrong number of coefficients
    with pytest.raises(ValueError):
        byte_decode(4, bytes(127))  # 32 * 4 = 128 bytes expected
