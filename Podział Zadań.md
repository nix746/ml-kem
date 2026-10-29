## Faza 1 (prototyp): zadania

**Cel:** `Decaps(Encaps(ek)) == K` dla 512 / 768 / 1024. Termin: ok. 5.11.  
**W prototypie:** wielomiany mnożymy zwykłą metodą, bez NTT.

### Osoba 1: algebra 

- `params.py`: stałe (`N`, `Q`, `ZETA`, `N_INV`), `MLKEMParams`, trzy zestawy, rozmiary kluczy.
- `crypto_primitives.py`: `h_hash`, `g_hash`, `j_hash`, `prf`, `xof` z `hashlib`.
- `polynomial.py`:
    - `poly_add`, `poly_sub`
    - `schoolbook_mul` (zwykłe mnożenie z zawijaniem `X²⁵⁶ = -1`)
    - operacje na wektorach i macierzach (`vec_add`, `matvec`, `dot`, transpozycja indeksów)
    - tablice `ZETAS` i `GAMMAS`, a jeśli starczy czasu `ntt` (forward)
- `tests/conftest.py`: adaptery do kyber-py i fixtury.
- **Testy:** hashe zgodne z `hashlib` i kyber-py, `schoolbook_mul` na prostych przypadkach (np. `X · X²⁵⁵ = -1`).

### Osoba 2: losowanie i kodowanie 

- `sampling.py`: `sample_ntt` (kolejność bajtów `j`, potem `i`) i `sample_poly_cbd`.
- `compression.py`: `compress`, `decompress`, `byte_encode`, `byte_decode` (plus pomocnicze `bits_to_bytes`, `bytes_to_bits`).
- `docs/CONTRACT.md`, przypięcie `kyber-py==1.2.0`, `.gitattributes`.
- **Testy:** kompresja dla wszystkich `x` od 0 do 3328 vs kyber-py, kodowanie bajtowe w obie strony, losowanie vs kyber-py.

### Osoba 3: szyfrowanie i wymiana klucza 

- `pke.py`: `k_pke_keygen`, `k_pke_encrypt`, `k_pke_decrypt` (wspólny licznik `N` dla `s, e` oraz `y, e1, e2`).
- `kem.py` w wersji podstawowej: `ml_kem_keygen`, `ml_kem_encaps`, `ml_kem_decaps` (ciche odrzucenie przez `J(z ‖ c)`, bez sprawdzania wejść).
- **Testy:** `Decrypt(Encrypt(m)) == m` oraz `Decaps(Encaps(ek)) == K` dla wszystkich trzech wariantów.
- Do czasu, aż osoby 1 i 2 oddadzą moduły, pisze według kolejności FIPS Alg. 13–21 na uzgodnionych schematach.