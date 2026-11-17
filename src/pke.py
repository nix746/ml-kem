"""Internal K-PKE encryption scheme (IND-CPA) as specified in FIPS 203 (Section 5)."""

import numpy as np

from compress import byte_decode, byte_encode, compress, decompress
from crypto_primitives import g_hash, prf
from params import MLKEMParams, N
from polynomial import (
    dot,
    inv_ntt,
    inv_ntt_vec,
    matvec,
    ntt_vec,
    poly_add,
    poly_sub,
    transpose,
)
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

    s_hat = ntt_vec(s)
    e_hat = ntt_vec(e)
    t_hat = poly_add(matvec(a_mat, s_hat), e_hat)

    ek_pke = b"".join(byte_encode(12, t_hat[i]) for i in range(k)) + rho
    dk_pke = b"".join(byte_encode(12, s_hat[i]) for i in range(k))
    return ek_pke, dk_pke


def k_pke_encrypt(params: MLKEMParams, ek_pke: bytes, m: bytes, r: bytes) -> bytes:
    """Encrypts a 32-byte message m using randomness r (Algorithm 14)."""
    k = params.k

    n_counter = 0

    t_hat = np.array(
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

    y_hat = ntt_vec(y)
    u = poly_add(inv_ntt_vec(matvec(transpose(a_mat), y_hat)), e1)

    mu = decompress(1, byte_decode(1, m))
    v = poly_add(poly_add(inv_ntt(dot(t_hat, y_hat)), e2), mu)

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

    s_hat = np.array(
        [byte_decode(12, dk_pke[384 * i : 384 * (i + 1)]) for i in range(k)],
        dtype=np.int64,
    )

    u_hat = ntt_vec(u)
    w = poly_sub(v, inv_ntt(dot(s_hat, u_hat)))

    return byte_encode(1, compress(1, w))
