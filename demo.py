"""Terminal walkthrough of one ML-KEM key exchange, stage by stage.

The six functions below are copies of the functions in src/pke.py and src/kem.py:
    k_pke_keygen, k_pke_encrypt, k_pke_decrypt              (src/pke.py)
    ml_kem_keygen_internal, ml_kem_encaps_internal,
    ml_kem_decaps_internal                                  (src/kem.py)

Usage:
    python demo.py                 ML-KEM-512 with random seeds
    python demo.py 768 --fixed     ML-KEM-768 with fixed seeds (same output every run)
    python demo.py --pause         wait for Enter between stages (for a live presentation)
"""

import argparse
import secrets
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

import kem  # noqa: E402
from compress import byte_decode, byte_encode, compress, decompress  # noqa: E402
from crypto_primitives import g_hash, h_hash, j_hash, prf  # noqa: E402
from params import ML_KEM_512, ML_KEM_768, ML_KEM_1024, MLKEMParams, N, Q  # noqa: E402
from polynomial import (  # noqa: E402
    dot,
    inv_ntt,
    inv_ntt_vec,
    matvec,
    ntt_vec,
    poly_add,
    poly_sub,
    transpose,
)
from sampling import sample_ntt, sample_poly_cbd  # noqa: E402

VARIANTS = {"512": ML_KEM_512, "768": ML_KEM_768, "1024": ML_KEM_1024}
PAUSE = False




def stage(title):
    """Prints a stage banner and optionally waits for Enter."""
    if PAUSE:
        input("\n   [Enter] ")
    print("\n" + "*" * 78)
    print(" " + title)
    print("*" * 78)


def step(text):
    print(f"\n-- {text}")


def show_bytes(label, b, n=16):
    """Prints the first n bytes in hex and the total length."""
    more = " ..." if len(b) > n else ""
    print(f"   {label:<14} {b[:n].hex(' ')}{more}  ({len(b)} bytes)")


def signed(f):
    """Shows coefficients as small signed numbers: 3328 -> -1, 3327 -> -2, ..."""
    f = np.asarray(f)
    return np.where(f > Q // 2, f - Q, f)


def show_poly(label, f, n=10, small=False):
    """Prints the first n coefficients of a polynomial (small=True prints them signed)."""
    values = signed(f) if small else np.asarray(f)
    text = " ".join(f"{int(x):5d}" for x in values[:n])
    print(f"   {label:<14} [{text} ...]  (256 coefficients)")


def show_counts(label, f):
    """Prints how many coefficients take each small value, e.g. -2: 21, -1: 63, ..."""
    values, counts = np.unique(signed(f), return_counts=True)
    text = ", ".join(f"{int(v):+d}: {int(c)}" for v, c in zip(values, counts, strict=True))
    print(f"   {label:<14} {text}")


def show_bits(label, bits, n=10):
    print(f"   {label:<14} [{' '.join(f'{int(b):5d}' for b in bits[:n])} ...]")


def k_pke_keygen(params: MLKEMParams, d: bytes, show=True) -> tuple[bytes, bytes]:
    """Generates the encryption key ek_pke and decryption key dk_pke (Algorithm 13)."""
    k = params.k

    rho, sigma = g_hash(d + bytes([k]))
    n_counter = 0
    if show:
        step("1. Split the seed: (rho, sigma) = G(d || k)")
        show_bytes("rho (public)", rho)
        show_bytes("sigma (secret)", sigma)

    a_mat = np.empty((k, k, N), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            a_mat[i, j] = sample_ntt(rho, i, j)
    if show:
        step(f"2. Expand rho into the public {k}x{k} matrix A_hat (SampleNTT)")
        show_poly("A_hat[0][0]", a_mat[0, 0])
        print("   (uniform numbers 0..3328; anyone who has rho gets the same matrix)")

    s = np.empty((k, N), dtype=np.int64)
    for i in range(k):
        s[i] = sample_poly_cbd(params.eta1, prf(params.eta1, sigma, n_counter))
        n_counter += 1

    e = np.empty((k, N), dtype=np.int64)
    for i in range(k):
        e[i] = sample_poly_cbd(params.eta1, prf(params.eta1, sigma, n_counter))
        n_counter += 1
    if show:
        step(f"3. Secret s and error e from sigma (CBD, eta1 = {params.eta1})")
        show_poly("s[0]", s[0], small=True)
        show_poly("e[0]", e[0], small=True)
        show_counts("values in s[0]", s[0])
        print("   (stored mod Q, so -1 is 3328 in the array; shown here with signs)")

    s_hat = ntt_vec(s)
    e_hat = ntt_vec(e)
    t_hat = poly_add(matvec(a_mat, s_hat), e_hat)
    if show:
        step("4. NTT of s and e, then t_hat = A_hat * s_hat + e_hat")
        show_poly("s_hat[0]", s_hat[0])
        show_poly("t_hat[0]", t_hat[0])
        print("   (t looks uniformly random: getting s back from (A, t) is Module-LWE)")

    ek_pke = b"".join(byte_encode(12, t_hat[i]) for i in range(k)) + rho
    dk_pke = b"".join(byte_encode(12, s_hat[i]) for i in range(k))
    if show:
        step("5. ByteEncode_12: 12 bits per coefficient")
        show_bytes("ek_pke", ek_pke)
        show_bytes("dk_pke", dk_pke)
        print(f"   ek_pke = {k} x 384 bytes of t_hat + 32 bytes of rho")
    return ek_pke, dk_pke


def k_pke_encrypt(params: MLKEMParams, ek_pke: bytes, m: bytes, r: bytes, show=True) -> bytes:
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
    if show:
        step("1. Read t_hat and rho from ek and rebuild the same matrix A_hat")
        show_poly("t_hat[0]", t_hat[0])
        show_poly("A_hat[0][0]", a_mat[0, 0])

    y = np.empty((k, N), dtype=np.int64)
    for i in range(k):
        y[i] = sample_poly_cbd(params.eta1, prf(params.eta1, r, n_counter))
        n_counter += 1

    e1 = np.empty((k, N), dtype=np.int64)
    for i in range(k):
        e1[i] = sample_poly_cbd(params.eta2, prf(params.eta2, r, n_counter))
        n_counter += 1

    e2 = sample_poly_cbd(params.eta2, prf(params.eta2, r, n_counter))
    if show:
        step("2. Fresh noise y, e1, e2 from r")
        show_poly("y[0]", y[0], small=True)
        show_poly("e1[0]", e1[0], small=True)
        show_poly("e2", e2, small=True)

    y_hat = ntt_vec(y)
    u = poly_add(inv_ntt_vec(matvec(transpose(a_mat), y_hat)), e1)
    if show:
        step("3. u = A^T * y + e1   (multiply in the NTT domain, then inverse NTT)")
        show_poly("u[0]", u[0])

    mu = decompress(1, byte_decode(1, m))
    v = poly_add(poly_add(inv_ntt(dot(t_hat, y_hat)), e2), mu)
    if show:
        step("4. Message as a polynomial (bit 0 -> 0, bit 1 -> 1665), then v = t^T*y + e2 + mu")
        show_bits("message bits", byte_decode(1, m))
        show_poly("mu", mu)
        show_poly("v", v)
        print("   (the message is hidden under t^T * y, which looks random)")

    c1 = b"".join(byte_encode(params.du, compress(params.du, u[i])) for i in range(k))
    c2 = byte_encode(params.dv, compress(params.dv, v))
    if show:
        step(f"5. Compress u to {params.du} bits and v to {params.dv} bits, then serialize")
        show_poly("u[0]", u[0])
        show_poly("compressed", compress(params.du, u[0]))
        show_poly("decompressed", decompress(params.du, compress(params.du, u[0])))
        print("   (compression loses a little precision; it acts like extra noise)")
        show_bytes("c1 (from u)", c1)
        show_bytes("c2 (from v)", c2)
    return c1 + c2


def k_pke_decrypt(params: MLKEMParams, dk_pke: bytes, c: bytes, show=True) -> bytes:
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
    if show:
        step("1. Decompress u and v from c, read s_hat from dk")
        show_poly("u[0]", u[0])
        show_poly("v", v)

    u_hat = ntt_vec(u)
    w = poly_sub(v, inv_ntt(dot(s_hat, u_hat)))
    if show:
        step("2. w = v - s^T * u   (the big random part cancels; message + small noise is left)")
        show_poly("w", w)
        distance = np.minimum(np.abs(signed(w)), np.abs(signed((w - 1665) % Q)))
        print(
            f"   every coefficient is within {int(distance.max())} of 0 or 1665;"
            f" decoding goes wrong only above {Q // 4}"
        )

    m_prime = byte_encode(1, compress(1, w))
    if show:
        step("3. Round: closer to 0 -> bit 0, closer to 1665 -> bit 1")
        show_bits("decoded bits", byte_decode(1, m_prime))
    return m_prime


def ml_kem_keygen_internal(params: MLKEMParams, d: bytes, z: bytes) -> tuple[bytes, bytes]:
    """Generates the ML-KEM key pair from seeds d and z (Algorithm 16)."""
    ek_pke, dk_pke = k_pke_keygen(params, d)
    ek = ek_pke
    dk = dk_pke + ek + h_hash(ek) + z
    step("6. ek = ek_pke,  dk = dk_pke || ek || H(ek) || z")
    show_bytes("ek (public)", ek)
    show_bytes("dk (secret)", dk)
    return ek, dk


def ml_kem_encaps_internal(params: MLKEMParams, ek: bytes, m: bytes) -> tuple[bytes, bytes]:
    """Encapsulates a shared secret using message randomness m (Algorithm 17)."""
    k_shared, r = g_hash(m + h_hash(ek))
    step("0. (K, r) = G(m || H(ek))")
    show_bytes("m", m)
    show_bytes("K (shared key)", k_shared)
    show_bytes("r", r)
    print("   (the encryption randomness r comes from m; Alice will need this to re-check c)")
    c = k_pke_encrypt(params, ek, m, r)
    return k_shared, c


def ml_kem_decaps_internal(params: MLKEMParams, dk: bytes, c: bytes) -> bytes:
    """Decapsulates ciphertext c with FO re-encryption check (Algorithm 18)."""
    k = params.k
    dk_pke = dk[: 384 * k]
    ek_pke = dk[384 * k : 768 * k + 32]
    h = dk[768 * k + 32 : 768 * k + 64]
    z = dk[768 * k + 64 : 768 * k + 96]

    m_prime = k_pke_decrypt(params, dk_pke, c)
    k_prime, r_prime = g_hash(m_prime + h)
    k_bar = j_hash(z + c)
    c_prime = k_pke_encrypt(params, ek_pke, m_prime, r_prime, show=False)
    step("4. (K', r') = G(m' || h), K_bar = J(z || c), encrypt m' again with r' -> c'")
    show_bytes("m'", m_prime)
    show_bytes("K'", k_prime)
    show_bytes("K_bar", k_bar)
    print(f"   c' == c ? {secrets.compare_digest(c, c_prime)}  -> return K' if equal, else K_bar")

    return k_prime if secrets.compare_digest(c, c_prime) else k_bar



def main():
    global PAUSE
    parser = argparse.ArgumentParser(description="Show one ML-KEM key exchange step by step.")
    parser.add_argument("variant", nargs="?", default="512", choices=VARIANTS)
    parser.add_argument("--fixed", action="store_true", help="use fixed seeds instead of random")
    parser.add_argument("--pause", action="store_true", help="wait for Enter between stages")
    args = parser.parse_args()
    PAUSE = args.pause
    params = VARIANTS[args.variant]

    if args.fixed:
        d, z, m = bytes(range(32)), bytes(range(32, 64)), bytes(range(64, 96))
    else:
        d, z, m = secrets.token_bytes(32), secrets.token_bytes(32), secrets.token_bytes(32)

    print(
        f"{params.name}: k={params.k}, eta1={params.eta1}, eta2={params.eta2},"
        f" du={params.du}, dv={params.dv}, Q={Q}"
    )
    print(f"sizes: ek {params.ek_size} B, dk {params.dk_size} B, ciphertext {params.ct_size} B")

    stage("ALICE: KeyGen  (Alg. 16 -> K-PKE.KeyGen, Alg. 13)")
    step("0. Random seeds d and z")
    show_bytes("d", d)
    show_bytes("z", z)
    ek, dk = ml_kem_keygen_internal(params, d, z)
    print("   Alice sends ek to Bob and keeps dk.")

    stage("BOB: Encaps  (Alg. 17 -> K-PKE.Encrypt, Alg. 14)")
    key_bob, c = ml_kem_encaps_internal(params, ek, m)
    print("   Bob sends c to Alice and keeps K.")

    stage("ALICE: Decaps  (Alg. 18 -> K-PKE.Decrypt, Alg. 15, + re-encryption)")
    key_alice = ml_kem_decaps_internal(params, dk, c)

    stage("RESULT")
    show_bytes("K (Bob)", key_bob, n=32)
    show_bytes("K (Alice)", key_alice, n=32)
    print(f"   same shared key: {key_bob == key_alice}   (an eavesdropper saw only ek and c)")

    stage("ATTACK: flip one bit of the ciphertext and give it to Alice")
    bad_c = bytes([c[0] ^ 1]) + c[1:]
    show_bytes("c (modified)", bad_c)
    key_bad = ml_kem_decaps_internal(params, dk, bad_c)
    print(f"   Alice's key == Bob's key ? {key_bad == key_bob}")
    print("   (no error message: the attacker learns nothing from the answer)")

    ek2, dk2 = kem.ml_kem_keygen_internal(params, d, z)
    key2, c2 = kem.ml_kem_encaps_internal(params, ek2, m)
    print(
        f"   src/kem.py gives the same ek: {ek2 == ek}, dk: {dk2 == dk}, c: {c2 == c},"
        f" K: {key2 == key_bob}"
    )
    print(
        f"   src/kem.py decaps of the modified c gives the same key: "
        f"{kem.ml_kem_decaps_internal(params, dk2, bad_c) == key_bad}"
    )
    try:
        from kyber_py.ml_kem.default_parameters import DEFAULT_PARAMETERS
        from kyber_py.ml_kem.ml_kem import ML_KEM
    except ImportError:
        print("   kyber-py not installed: skipping the comparison with the reference")
        return
    oracle = ML_KEM(DEFAULT_PARAMETERS[params.name.replace("ML-KEM-", "ML")])
    ek_ref, dk_ref = oracle._keygen_internal(d, z)
    key_ref, c_ref = oracle._encaps_internal(ek_ref, m)
    print(
        f"   kyber-py (reference) gives the same ek: {ek_ref == ek}, dk: {dk_ref == dk},"
        f" c: {c_ref == c}, K: {key_ref == key_bob}"
    )


if __name__ == "__main__":
    main()
