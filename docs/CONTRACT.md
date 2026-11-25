# Interface Contract (DRAFT)
 
This document fixes the conventions that cross module boundaries: how polynomials are
represented, which value ranges functions accept and return, byte and bit orders, and the exact
signatures of the functions we will call. Each convention below states **what** we
need to agree on and **why**, with references to NIST FIPS 203. Please confirm those thesises and assumptions.
 
---
 
## 1. Notation and constants
 
| Symbol | Value | Meaning |
|---|---|---|
| `N` | 256 | Number of coefficients of a polynomial in `R_q = Z_q[X]/(X^256 + 1)` |
| `Q` | 3329 | Prime modulus. `Q - 1 = 3328 = 2^8 * 13`, so a primitive 256-th root of unity exists mod `Q` (`zeta = 17`) |
| `ZETA` | 17 | Primitive 256-th root of unity mod `Q`: `17^256 = 1` and `17^128 = Q - 1` (FIPS 203, Section 4.3) |
| `N_INV` | 3303 | `128^-1 mod Q`, the final factor of `inv_ntt` (the NTT has 7 layers, so it is `128^-1`, not `256^-1`) |
 
Parameter sets (FIPS 203, Section 8, Table 2):
 
| Set | `k` | `eta1` | `eta2` | `du` | `dv` |
|---|---|---|---|---|---|
| ML-KEM-512 | 2 | 3 | 2 | 10 | 4 |
| ML-KEM-768 | 3 | 2 | 2 | 10 | 4 |
| ML-KEM-1024 | 4 | 2 | 2 | 11 | 5 |
 
`N`, `Q`, `ZETA` and `N_INV` live in `src/params.py` and are imported from there; no module
hard-codes `3329`.
 
---
 
## 2. Data representation (all modules)
 
### 2.1 Polynomials
 
A polynomial `f = f_0 + f_1 X + ... + f_255 X^255` is a NumPy array:
 
- **shape** `(256,)`, index `i` holds the coefficient of `X^i`;
- **dtype** `np.int64`;
- **values** in the canonical range `[0, Q)` on every function **output**.
*Why `int64` and not `int32`:* a single product of two coefficients is below `3329^2 ≈ 1.1 * 10^7`,
which fits in `int32`, but schoolbook multiplication and the matrix-vector product `Â ∘ ŝ` sum several
such products before reducing (`schoolbook_mul` reaches `256 * 3328^2 ≈ 2.8 * 10^9 > 2^31`). With `int64` there is no risk of silent overflow (NumPy does not raise on
integer overflow, it wraps around), and the performance cost is irrelevant for this project.
 
*Why `[0, Q)` and not signed values:* FIPS 203 defines every intermediate value as an element of
`Z_q`, and the reference oracle `kyber-py` stores coefficients the same way. A single canonical
range means outputs can be compared with `np.array_equal` directly. In particular, a noise
coefficient `-1` from CBD is returned as `3328`.
 
### 2.2 Vectors and matrices of polynomials
 
- A vector in `R_q^k` is an array of shape `(k, 256)`.
- A matrix in `R_q^{k x k}` is an array of shape `(k, k, 256)`, where `A[i, j]` is the polynomial in
  row `i`, column `j`.
*Why:* NumPy broadcasting then works over the leading axes, e.g. `compress(du, u)` compresses the
whole vector `u` at once.
 
### 2.3 NTT domain
 
Arrays do not carry a flag saying whether they are in the normal or the NTT domain. By convention,
variable names with a `_hat` suffix (`a_hat`, `s_hat`) are in the NTT domain, matching the `^`
notation in FIPS 203. Note that `sample_ntt` returns a polynomial **already in the NTT domain**.
Which functions expect and return which domain is listed in Section 7.
 
### 2.4 Bytes and bits
 
- Byte strings are Python `bytes` (not NumPy arrays) at every public interface.
- Bit order is **little-endian within each byte**: bit `0` of the bit array is the least
  significant bit of byte `0`; bit `8` is the least significant bit of byte `1` (FIPS 203,
  Algorithms 3 and 4).
- A single byte argument of the hash functions (e.g. `i`, `j`, `b`) is passed as a Python `int` in
  `[0, 255]` and converted with `bytes([x])` inside the function.
---
 
## 3. Error handling
 
Public functions raise `ValueError` when an argument has the wrong length, shape or an unsupported
`d` / `eta`. They do **not** silently reduce or truncate.
 
*Why:* an off-by-one in a length or a swapped argument should fail immediately at the point of the
mistake, not three modules later as a "wrong shared key". The checks cost a few comparisons.
 
One deliberate exception is `byte_decode(12, ...)`, which reduces modulo `Q` (see 4.2), because the
standard requires it.
 
---
 
## 4. `src/compress.py` (owner: Person 2)

The module was called `compression.py` at first. Python 3.14 ships a standard-library package with
that name, which pytest imports before our `src/` directory is on the path, so `pke.py` received the
standard-library module instead of ours. It was renamed to `compress.py`.
 
### 4.1 `compress(d: int, x: np.ndarray) -> np.ndarray`
### `decompress(d: int, y: np.ndarray) -> np.ndarray`
 
FIPS 203, Section 4.2.1, equations (4.7) and (4.8):
 
```
Compress_d(x)   = round((2^d / Q) * x) mod 2^d
Decompress_d(y) = round((Q / 2^d) * y)
```
 
- Accepted `d`: `1 <= d < 12`. In ML-KEM only `d in {1, du, dv} = {1, 4, 5, 10, 11}` occur.
- Input of `compress`: any shape, values in `[0, Q)`. Output: same shape, values in `[0, 2^d)`.
- Input of `decompress`: any shape, values in `[0, 2^d)`. Output: same shape, values in `[0, Q)`.
- **Integer arithmetic only**, no floats:
  - `compress`:   `((x << d) + Q // 2) // Q` then `& (2^d - 1)`
  - `decompress`: `(Q * y + 2^(d-1)) >> d`
*Why integers:* `2^d / Q` is not exactly representable as a float; for some `x` the float result
lands on the wrong side of `.5` and rounds incorrectly. With integers the result is exact. Because
`Q` is odd, an exact tie `.5` never occurs, so "round half up" is unambiguous.
 
*Mathematical property (tested):* compression loses information, but the error is bounded:
 
```
|x - Decompress_d(Compress_d(x))| mod± Q  <=  round(Q / 2^(d+1))
```
 
| `d` | 1 | 4 | 5 | 10 | 11 |
|---|---|---|---|---|---|
| bound | 832 | 104 | 52 | 2 | 1 |
 
These bounds are tight (verified for every `x` in `[0, Q)`). This compression error is added to
the LWE noise and is part of the decryption-failure analysis. For `d = 1`, `Compress_1` is exactly
the message decoder used in `K-PKE.Decrypt`: it returns `1` when `x` is closer to `Q/2` than to `0`.
 
### 4.2 `byte_encode(d: int, f: np.ndarray) -> bytes`
### `byte_decode(d: int, b: bytes) -> np.ndarray`
 
FIPS 203, Algorithms 5 and 6.
 
- Accepted `d`: `1 <= d <= 12`.
- `byte_encode`: `f` has shape `(256,)` with values in `[0, m)`, where `m = 2^d` for `d < 12` and
  `m = Q` for `d = 12`. Output length: `32 * d` bytes.
- `byte_decode`: `b` has length exactly `32 * d`. Output: shape `(256,)`.
  - For `d < 12` the output values are in `[0, 2^d)`.
  - For `d = 12` each 12-bit value (up to 4095) is **reduced modulo `Q`**, so the output is in
    `[0, Q)`. This is what Algorithm 6 specifies, and it is what makes the encapsulation-key
    "modulus check" (FIPS 203, Section 7.2) work:
    `byte_encode(12, byte_decode(12, ek_slice)) == ek_slice` fails exactly when `ek` contains a
    coefficient `>= Q`.
- Each coefficient occupies `d` consecutive bits, least significant bit first, coefficients in
  order `f_0, f_1, ...` (see 2.4).
Vectors are encoded by concatenating the encodings of their polynomials in order; that is done by
the caller (`pke.py`), not here.
 
### 4.3 Helpers: `bits_to_bytes(bits) -> bytes`, `bytes_to_bits(b) -> np.ndarray`
 
FIPS 203, Algorithms 3 and 4. The bit array is a NumPy array of `0`/`1` values with length a
multiple of 8. Both are public because `sampling.py` reuses `bytes_to_bits`.
 
---
 
## 5. `src/sampling.py` (owner: Person 2)
 
### 5.1 `sample_ntt(rho: bytes, i: int, j: int) -> np.ndarray`
 
FIPS 203, Algorithm 7. Returns **the entry `Â[i, j]`** of the public matrix, in the NTT domain,
shape `(256,)`, values in `[0, Q)`.
 
- `rho`: exactly 32 bytes. `i`, `j`: integers in `[0, k)`.
- **Byte order:** the function feeds the XOF with `rho || j || i` (first `j`, then `i`), as in
  FIPS 203, Algorithm 13 line 6 and Algorithm 14 line 6.
*Why the swap is hidden inside the function:* the caller writes the natural
`A_hat[i, j] = sample_ntt(rho, i, j)` and cannot get the order wrong. If the order were swapped,
our implementation would still decrypt its own ciphertexts (it would just use `Â^T` consistently),
so only a comparison with `kyber-py` detects the bug. That is why there is a dedicated test that
compares the whole matrix.
 
Algorithm (rejection sampling): read the XOF output 3 bytes `b0, b1, b2` at a time and form two
12-bit candidates
 
```
d1 = b0 + 256 * (b1 mod 16)
d2 = floor(b1 / 16) + 16 * b2
```
 
keeping each one if it is `< Q`, in the order `d1, d2`, until 256 coefficients are collected.
Each candidate is uniform on `[0, 4096)`, so an accepted value is uniform on `[0, Q)`. The
acceptance probability is `3329 / 4096 ≈ 81.3 %`; on average about 473 bytes are needed.
 
Running out of XOF output must not be possible: if the bytes generated so far are not enough, more
are requested (SHAKE output of length `2L` starts with the output of length `L`, so this does not
change the result).
 
### 5.2 `sample_poly_cbd(eta: int, buf: bytes) -> np.ndarray`
 
FIPS 203, Algorithm 8.
 
- `eta in {2, 3}`; `buf` has length exactly `64 * eta` (the output length of `PRF_eta`).
- Output: shape `(256,)`, values in `[0, Q)` (a coefficient `-e` is stored as `Q - e`).
For coefficient `i`, take the `2 * eta` bits `bits[2*eta*i : 2*eta*(i+1)]`; the result is
(sum of the first `eta` bits) minus (sum of the next `eta` bits), reduced mod `Q`.
 
*Mathematical property (tested statistically):* each coefficient is the difference of two
independent `Binomial(eta, 1/2)` variables, so it lies in `[-eta, eta]`, has mean `0` and variance
`eta / 2`. This is the "small error" of Module-LWE.
 
---
 
## 6. `src/crypto_primitives.py` (owner: Person 1)

FIPS 203, Section 4.1. Thin wrappers around `hashlib`; all inputs and outputs are `bytes`.

| Function | Signature | Definition |
|---|---|---|
| `h_hash` | `(data: bytes) -> bytes` | `SHA3-256(data)`, 32 bytes |
| `j_hash` | `(data: bytes) -> bytes` | `SHAKE256(data, 32 bytes)` |
| `g_hash` | `(data: bytes) -> tuple[bytes, bytes]` | `SHA3-512(data)` split into two 32-byte halves |
| `prf` | `(eta: int, s: bytes, b: int) -> bytes` | `SHAKE256(s \|\| bytes([b]), 64 * eta bytes)` |
| `xof` | `(seed: bytes, length: int) -> bytes` | `SHAKE128(seed)` truncated to `length` bytes |

- `prf` raises `ValueError` unless `eta in {2, 3}`, `len(s) == 32` and `0 <= b <= 255`.
- `xof` raises `ValueError` for a negative `length`.
- `g_hash` is called with `d + bytes([k])` in K-PKE.KeyGen (FIPS 203, Algorithm 13, line 1), not
  with `d` alone. The extra byte `k` was added in the final FIPS 203; without it the keys still
  work with our own code but differ from every other implementation.

*Why `xof(seed, length)` and not a stream:* FIPS 203 describes the XOF as a stream (`Init`,
`Absorb`, `Squeeze`), but `hashlib.shake_128` returns a fixed number of bytes per call. A caller that
needs more bytes asks again with a larger `length`; the result starts with the shorter one, so the
first bytes do not change. `xof` knows nothing about the indices `i`, `j`: `sample_ntt` builds
`seed = rho + bytes([j, i])` itself, which keeps the index order in one place (5.1).

---

## 7. `src/polynomial.py` (owner: Person 1)

FIPS 203, Sections 2.4 and 4.3. Shapes, dtypes and value ranges follow Section 2:
a polynomial is `(256,)`, a vector `(k, 256)`, a matrix `(k, k, 256)`, all `np.int64`, and every
**output** is in `[0, Q)`. Inputs are not range-checked: any integers are accepted and reduced
mod `Q`. No function modifies its arguments.

### 7.1 Which domain each function works in

| Function | Input | Output | Domain | FIPS 203 |
|---|---|---|---|---|
| `poly_add(a, b)`, `poly_sub(a, b)` | equal shapes ending in 256 | same shape | either (the NTT is linear) | Section 2.4.5 |
| `schoolbook_mul(a, b)` | `(256,)`, `(256,)` | `(256,)` | normal | none (test reference) |
| `ntt(f)` | `(256,)` | `(256,)` | normal → NTT | Algorithm 9 |
| `inv_ntt(f_hat)` | `(256,)` | `(256,)` | NTT → normal | Algorithm 10 |
| `base_case_multiply(a0, a1, b0, b1, gamma)` | five `int` | `tuple[int, int]` | one NTT pair | Algorithm 12 |
| `multiply_ntts(f_hat, g_hat)` | `(256,)`, `(256,)` | `(256,)` | NTT | Algorithm 11 |
| `ntt_vec(v)` | `(k, 256)` | `(k, 256)` | normal → NTT | — |
| `inv_ntt_vec(v_hat)` | `(k, 256)` | `(k, 256)` | NTT → normal | — |
| `transpose(a_hat)` | `(k, k, 256)` | `(k, k, 256)` | either | — |
| `dot(u_hat, v_hat)` | `(k, 256)`, `(k, 256)` | `(256,)` | NTT | Section 2.4.7 |
| `matvec(a_hat, v_hat)` | `(k, k, 256)`, `(k, 256)` | `(k, 256)` | NTT | Section 2.4.7 |

**`dot` and `matvec` multiply with `multiply_ntts`, so both arguments must already be in the NTT
domain and the result is in the NTT domain.** Passing normal-domain polynomials raises no error; it
just gives a meaningless result. This is why NTT-domain variables carry the `_hat` suffix (2.3).

### 7.2 Semantics

- `poly_add` and `poly_sub` work coefficient-wise, so they also accept vectors `(k, 256)`; there is
  no separate `vec_add`.
- `schoolbook_mul` computes the full product of degree `<= 510` and wraps it with `X^256 = -1`:
  `(c[:256] - c[256:]) % Q`. It is slow (65 536 coefficient products) and exists as an independent
  reference for the NTT tests.
- `ntt(f)` returns 128 pairs: positions `2i` and `2i + 1` hold `f mod (X^2 - GAMMAS[i])`, as
  `a + b X`. Example: `ntt(X^2)` is `[GAMMAS[0], 0, GAMMAS[1], 0, ...]`.
- `inv_ntt(ntt(f)) == f` for every `f`, and
  `inv_ntt(multiply_ntts(ntt(f), ntt(g))) == schoolbook_mul(f, g)`.
- `matvec(a_hat, v_hat)[i] = dot(a_hat[i], v_hat)`. The transposed product `Â^T ∘ ŷ` of
  K-PKE.Encrypt is written `matvec(transpose(a_hat), y_hat)`.
- `transpose` swaps only the first two axes (`np.transpose(a, (1, 0, 2))`). Plain `a.T` would also
  reverse the coefficient axis.
- `*` and `@` on NumPy arrays do **not** multiply polynomials; use `schoolbook_mul`,
  `multiply_ntts`, `dot` or `matvec`.

### 7.3 Tables

```
ZETAS[i]  = 17^BitRev7(i)       mod Q    i = 0..127, used by ntt / inv_ntt
GAMMAS[i] = 17^(2*BitRev7(i)+1) mod Q    i = 0..127, used by multiply_ntts
```

`BitRev7(i)` reverses the 7 bits of `i` (e.g. `1 -> 64`). Both tables are computed at import and
match FIPS 203, Appendix A (`ZETAS` starts `1, 1729, 2580, 3289`, `GAMMAS` starts
`17, 3312, 2761, 568`). `ntt` uses `ZETAS[1]` to `ZETAS[127]`; `ZETAS[0] = 1` is never used.

### 7.4 Errors

`ValueError` is raised when:

- the two arguments of `poly_add`, `poly_sub`, `schoolbook_mul` or `multiply_ntts` have different
  shapes or a last axis other than 256,
- `schoolbook_mul`, `ntt`, `inv_ntt` or `multiply_ntts` get anything other than a single `(256,)`
  polynomial (this also applies to each row given to `ntt_vec` / `inv_ntt_vec`),
- `dot` gets vectors of different lengths (so `matvec` raises when the row length differs from
  the length of the vector).

### 7.5 Use in `pke.py` (target, Phase 2)

```python
s_hat = ntt_vec(s)
t_hat = poly_add(matvec(a_hat, s_hat), ntt_vec(e))  # KeyGen: ek encodes t_hat
y_hat = ntt_vec(y)
u = poly_add(inv_ntt_vec(matvec(transpose(a_hat), y_hat)), e1)  # Encrypt
v = poly_add(poly_add(inv_ntt(dot(t_hat, y_hat)), e2), mu)  # Encrypt
w = poly_sub(v, inv_ntt(dot(s_hat, ntt_vec(u))))  # Decrypt
```

`e1`, `e2` and `mu` are added in the normal domain, after `inv_ntt`. Since PR #9 (17.11.2026)
`pke.py` uses exactly these calls, and `test_k_pke_matches_kyber_py_oracle` and
`test_ml_kem_internal_deterministic_matches_oracle` confirm byte-for-byte agreement with kyber-py.

---

## 8. Testing conventions
 
- The reference oracle is `kyber-py==1.2.0`, pinned exactly in `requirements.txt` and
  `pyproject.toml`. Tests use some of its internal methods (`_compress_ele`, `ntt_sample`, `cbd`,
  `_generate_matrix_from_seed`, `_H`, `_G`, `_J`, `_prf`, `_xof`, `_k_pke_keygen`,
  `_keygen_internal`, `_encaps_internal`), which may change in other versions; the pin keeps tests
  stable.
- Shared helpers live in `tests/conftest.py`: `to_kyber`, `to_kyber_ntt` and `from_kyber` convert
  between our arrays and kyber-py polynomials, `random_poly(rng)` draws a polynomial in `[0, Q)`,
  and the fixture `rng` is a generator with a fixed seed.
- `kyber-py` is imported only under `tests/`, never in `src/`.
- Tests that compare with the oracle convert our arrays with `.tolist()` and compare with the
  oracle's `.coeffs` lists.
- Randomized tests use a fixed seed (`np.random.default_rng(<seed>)`) so a failure can be
  reproduced.