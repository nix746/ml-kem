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
 
Parameter sets (FIPS 203, Section 8, Table 2):
 
| Set | `k` | `eta1` | `eta2` | `du` | `dv` |
|---|---|---|---|---|---|
| ML-KEM-512 | 2 | 3 | 2 | 10 | 4 |
| ML-KEM-768 | 3 | 2 | 2 | 10 | 4 |
| ML-KEM-1024 | 4 | 2 | 2 | 11 | 5 |
 
`N` and `Q` live in `src/params.py` and are imported from there; no module hard-codes `3329`.
 
---
 
## 2. Data representation (all modules)
 
### 2.1 Polynomials
 
A polynomial `f = f_0 + f_1 X + ... + f_255 X^255` is a NumPy array:
 
- **shape** `(256,)`, index `i` holds the coefficient of `X^i`;
- **dtype** `np.int64`;
- **values** in the canonical range `[0, Q)` on every function **output**.
*Why `int64` and not `int32`:* a single product of two coefficients is below `3329^2 ≈ 1.1 * 10^7`,
which fits in `int32`, but NTT multiplication and the matrix-vector product `Â ∘ ŝ` sum several such
products before reducing. With `int64` there is no risk of silent overflow (NumPy does not raise on
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
 
## 4. `src/compression.py` (owner: Person 2)
 
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
 
## 6. `src/crypto_primitives.py` [OPEN: owner to confirm]
 
`docs/project_notes.md` assigns this file to Person 2, while the current task list does not.
`sample_ntt` and `pke.py` depend on it, so the signatures must be fixed:
 
| Function | Signature | Definition (FIPS 203, Section 4.1) |
|---|---|---|
| `h_hash` | `(data: bytes) -> bytes` | `SHA3-256(data)`, 32 bytes |
| `j_hash` | `(data: bytes) -> bytes` | `SHAKE256(data, 32 bytes)` |
| `g_hash` | `(data: bytes) -> tuple[bytes, bytes]` | `SHA3-512(data)` split into two 32-byte halves |
| `prf` | `(eta: int, s: bytes, b: int) -> bytes` | `SHAKE256(s \|\| bytes([b]), 64 * eta bytes)`, `len(s) == 32` |
| `xof` | **[OPEN]** see below | `SHAKE128` |
 
**[OPEN] `xof`:** the current stub `xof(rho, i, j)` does not say what it returns or in which order
it concatenates `i` and `j`. Python's `hashlib.shake_128` is not a stream: it returns a fixed number
of bytes per call. Proposal:
 
```python
def xof(seed: bytes, length: int) -> bytes:
    """SHAKE128(seed) truncated to `length` bytes."""
```
 
with `sample_ntt` building `seed = rho + bytes([j, i])` itself. This keeps the index order in one
place (5.1) instead of two.
 
Until this is decided, `sample_ntt` may call `hashlib.shake_128` directly.
 
---
 
## 7. Testing conventions
 
- The reference oracle is `kyber-py==1.2.0`, pinned exactly in `requirements.txt` and
  `pyproject.toml`. Tests use some of its internal methods (`_compress_ele`, `ntt_sample`, `cbd`,
  `_generate_matrix_from_seed`), which may change in other versions; the pin keeps tests stable.
- `kyber-py` is imported only under `tests/`, never in `src/`.
- Tests that compare with the oracle convert our arrays with `.tolist()` and compare with the
  oracle's `.coeffs` lists.
- Randomized tests use a fixed seed (`np.random.default_rng(<seed>)`) so a failure can be
  reproduced.