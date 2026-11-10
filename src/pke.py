"""Internal K-PKE encryption scheme (IND-CPA) as specified in FIPS 203 (Section 5)."""

import numpy as np

from compression import byte_decode, byte_encode, compress, decompress
from crypto_primitives import g_hash, prf
from params import MLKEMParams, N
from polynomial import poly_add, poly_sub, schoolbook_mul
from sampling import sample_ntt, sample_poly_cbd

def k_pke_keygen(params: MLKEMParams, d: bytes) -> tuple[bytes, bytes]:
    """Generates the encryption key ek_pke and decryption key dk_pke (Algorithm 13)."""
    k = params.k

    rho, sigma = g_hash(d + bytes([k]))
    n_counter = 0

    a_mat = np.empty((k, k, N), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            a_mat[i, j] = sample_ntt(rho, i, j)

    s = np.empty((k, N), dtype=np.int64)
    for i in range(k):
        s[i] = sample_poly_cbd(params.eta1, prf(params.eta1, sigma, n_counter))
        n_counter += 1

    e = np.empty((k, N), dtype=np.int64)
    for i in range(k):
        e[i] = sample_poly_cbd(params.eta1, prf(params.eta1, sigma, n_counter))
        n_counter += 1

    # schoolbook multiplication to be changed for multiply_ntts
    t = e.copy()
    for i in range(k):
        for j in range(k):
            t[i] = poly_add(t[i], schoolbook_mul(a_mat[i, j], s[j]))

    ek_pke = b"".join(byte_encode(12, t[i]) for i in range(k)) + rho
    dk_pke = b"".join(byte_encode(12, s[i]) for i in range(k))
    return ek_pke, dk_pke


def k_pke_encrypt(params: MLKEMParams, ek_pke: bytes, m: bytes, r: bytes) -> bytes:
    """Encrypts a 32-byte message m using randomness r (Algorithm 14)."""
    k = params.k

    n_counter = 0

    t = np.array(
        [byte_decode(12, ek_pke[384 * i : 384 * (i + 1)]) for i in range(k)],
        dtype=np.int64,
    )

    rho = ek_pke[384 * k : 384 * k + 32]

    a_mat = np.empty((k, k, N), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            a_mat[i, j] = sample_ntt(rho, i, j)

    y = np.empty((k, N), dtype=np.int64)
    for i in range(k):
        y[i] = sample_poly_cbd(params.eta1, prf(params.eta1, r, n_counter))
        n_counter += 1

    e1 = np.empty((k, N), dtype=np.int64)
    for i in range(k):
        e1[i] = sample_poly_cbd(params.eta2, prf(params.eta2, r, n_counter))
        n_counter += 1

    e2 = sample_poly_cbd(params.eta2, prf(params.eta2, r, n_counter))

    u = e1.copy()
    for i in range(k):
        for j in range(k):
            u[i] = poly_add(u[i], schoolbook_mul(a_mat[j, i], y[j]))

    mu = decompress(1, byte_decode(1, m))

    # schoolbook multiplication to be changed for multiply_ntts
    v = poly_add(e2, mu)
    for i in range(k):
        v = poly_add(v, schoolbook_mul(t[i], y[i]))
        
    c1 = b"".join(byte_encode(params.du, compress(params.du, u[i])) for i in range(k))
    c2 = byte_encode(params.dv, compress(params.dv, v))
    return c1 + c2


def k_pke_decrypt(params: MLKEMParams, dk_pke: bytes, c: bytes) -> bytes:
    """Decrypts ciphertext c and returns the 32-byte message m' (Algorithm 15)."""
    k = params.k
    du_bytes = 32 * params.du

    c1 = c[: du_bytes * k]
    c2 = c[du_bytes * k :]

    u = np.array(
        [
            decompress(params.du, byte_decode(params.du, c1[du_bytes * i : du_bytes * (i + 1)]))
            for i in range(k)
        ],
        dtype=np.int64,
    )

    v = decompress(params.dv, byte_decode(params.dv, c2))

    s = np.array(
        [byte_decode(12, dk_pke[384 * i : 384 * (i + 1)]) for i in range(k)],
        dtype=np.int64,
    )

    w = v.copy()
    for i in range(k):
        w = poly_sub(w, schoolbook_mul(s[i], u[i]))

    return byte_encode(1, compress(1, w))
