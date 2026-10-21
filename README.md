# Post-Quantum Cryptography: Lattice-Based Foundations (ML-KEM)

Projekt grupowy implementujący od podstaw standard **NIST FIPS 203 (ML-KEM / CRYSTALS-Kyber)** w oparciu o problem **Module-LWE (Module Learning With Errors)**.

---

## Szybki start (Konfiguracja środowiska)

### Wariant 1: Użycie `uv` (Rekomendowane)

1. Zainstaluj zależności i utwórz środowisko `.venv`:
   ```powershell
   uv sync
   ```
2. Uruchom testy:
   ```powershell
   uv run pytest
   ```
3. Sprawdź styl i formatowanie kodu (`ruff`):
   ```powershell
   uv run ruff check --fix
   uv run ruff format
   ```

### Wariant 2: Klasyczny `venv` + `pip`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pytest ruff kyber-py
pytest
```

---

## Struktura projektu (`src/`)

Architektura odpowiada sekcjom oficjalnego standardu **NIST FIPS 203**:

```text
ml-kem/
├── .gitignore
├── pyproject.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── params.py            # Stałe (n=256, q=3329) i zestawy parametrów (512, 768, 1024)
│   ├── crypto_primitives.py # Funkcje skrótu i XOF ze standardu SHA-3 (H, G, J, PRF, XOF)
│   ├── polynomial.py        # Pierścień wielomianów R_q i szybka transformata NTT / InvNTT
│   ├── sampling.py          # Losowanie wielomianów (SampleNTT i rozkład dwumianowy SamplePolyCBD)
│   ├── compression.py       # Kompresja (Compress/Decompress) i kodowanie bajtowe (ByteEncode/Decode)
│   ├── pke.py               # Wewnętrzny schemat szyfrowania asymetrycznego K-PKE (KeyGen, Encrypt, Decrypt)
│   └── kem.py               # Pełny mechanizm ML-KEM z transformacją Fujisaki-Okamoto (KeyGen, Encaps, Decaps)
└── tests/
    └── test_primitives.py   # Testy środowiska i wyroczni referencyjnej
```

* [`src/params.py`](src/params.py) – Stałe globalne ($n=256$, $q=3329$) oraz zestawy parametrów dla `ML-KEM-512`, `ML-KEM-768` i `ML-KEM-1024` (FIPS 203 Sekcja 8).
* [`src/crypto_primitives.py`](src/crypto_primitives.py) – Funkcje skrótu i XOF ze standardu SHA-3 (`H`, `G`, `J`, `PRF`, `XOF` – FIPS 203 Sekcja 4.1).
* [`src/polynomial.py`](src/polynomial.py) – Pierścień wielomianów $R_q = \mathbb{Z}_q[X]/(X^{256}+1)$ oraz szybka transformata **NTT / InvNTT** (FIPS 203 Sekcja 4.3).
* [`src/sampling.py`](src/sampling.py) – Generowanie wielomianów z rozkładu jednostajnego (`SampleNTT`) oraz szumu z rozkładu dwumianowego (`SamplePolyCBD` – FIPS 203 Sekcja 4.2.2).
* [`src/compression.py`](src/compression.py) – Kompresja współczynników (`Compress` / `Decompress`) oraz serializacja bitowa (`ByteEncode` / `ByteDecode` – FIPS 203 Sekcja 4.2.1).
* [`src/pke.py`](src/pke.py) – Wewnętrzny schemat szyfrowania asymetrycznego `K-PKE` (`KeyGen`, `Encrypt`, `Decrypt` – FIPS 203 Sekcja 5).
* [`src/kem.py`](src/kem.py) – Pełny mechanizm `ML-KEM` z transformacją Fujisaki-Okamoto i cichym odrzuceniem (`KeyGen`, `Encaps`, `Decaps` – FIPS 203 Sekcja 6 i 7).

---

## Zasady pracy w zespole

1. W kodzie produkcyjnym (`src/`) korzystamy wyłącznie z `numpy` oraz biblioteki standardowej Pythona (`hashlib`, `os`, `secrets`).
2. Biblioteka `kyber-py` służy **wyłącznie w katalogu `tests/`** jako wyrocznia referencyjna (*Reference Oracle*) do porównywania wyników poszczególnych funkcji.
3. Przed każdym commitem uruchamiamy `uv run ruff format` oraz `uv run pytest`.
