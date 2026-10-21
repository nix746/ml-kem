# Post-Quantum Cryptography: Lattice-Based Foundations (ML-KEM)

A group project implementing the **NIST FIPS 203 (ML-KEM / CRYSTALS-Kyber)** standard from scratch, based on the **Module-LWE (Module Learning With Errors)** problem.

---

## Quick start (environment setup)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -v
ruff check
```

## Project structure (`src/`)

The architecture mirrors the sections of the official **NIST FIPS 203** standard:

```text
ml-kem/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── docs/
│   └── project_notes.md     # Libraries, design rationale and implementation roadmap
├── src/
│   ├── __init__.py
│   ├── params.py            # Constants (n=256, q=3329) and parameter sets (512, 768, 1024)
│   ├── crypto_primitives.py # SHA-3 based hash functions and XOF (H, G, J, PRF, XOF)
│   ├── polynomial.py        # Polynomial ring R_q and fast NTT / inverse NTT
│   ├── sampling.py          # Polynomial sampling (SampleNTT and SamplePolyCBD)
│   ├── compression.py       # Compress/Decompress and ByteEncode/ByteDecode
│   ├── pke.py               # Internal public-key encryption scheme K-PKE
│   └── kem.py               # Full ML-KEM with the Fujisaki-Okamoto transform
└── tests/
    └── test_primitives.py   # Environment and reference oracle tests
```

* [`src/params.py`](src/params.py) – Global constants ($n=256$, $q=3329$) and parameter sets for `ML-KEM-512`, `ML-KEM-768` and `ML-KEM-1024` (FIPS 203, Section 8).
* [`src/crypto_primitives.py`](src/crypto_primitives.py) – SHA-3 based hash functions and XOF (`H`, `G`, `J`, `PRF`, `XOF` – FIPS 203, Section 4.1).
* [`src/polynomial.py`](src/polynomial.py) – Polynomial ring $R_q = \mathbb{Z}_q[X]/(X^{256}+1)$ and the fast **NTT / inverse NTT** (FIPS 203, Section 4.3).
* [`src/sampling.py`](src/sampling.py) – Uniform sampling of polynomials (`SampleNTT`) and noise sampling from the centered binomial distribution (`SamplePolyCBD` – FIPS 203, Section 4.2.2).
* [`src/compression.py`](src/compression.py) – Coefficient compression (`Compress` / `Decompress`) and bit-level serialization (`ByteEncode` / `ByteDecode` – FIPS 203, Section 4.2.1).
* [`src/pke.py`](src/pke.py) – Internal public-key encryption scheme `K-PKE` (`KeyGen`, `Encrypt`, `Decrypt` – FIPS 203, Section 5).
* [`src/kem.py`](src/kem.py) – Full `ML-KEM` mechanism with the Fujisaki-Okamoto transform and implicit rejection (`KeyGen`, `Encaps`, `Decaps` – FIPS 203, Sections 6 and 7).

---

## Team guidelines

1. Production code (`src/`) uses only `numpy` and the Python standard library (`hashlib`, `os`, `secrets`).
2. The `kyber-py` library is used **only inside `tests/`** as a *reference oracle* to compare the outputs of individual functions.
3. Before every commit, run `ruff check`, `ruff format` and `pytest`.
