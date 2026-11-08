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


# Faza 2: NTT i zgodność z FIPS 203

**Cel:** zgodność bit w bit z kyber-py. Dla tych samych `d`, `z`, `m` wartości `ek`, `dk`, `c`, `K` są identyczne dla ML-KEM-512 / 768 / 1024.
**Termin:** ok. 11–12.11.
**Założenie:** faza 1 jest zakończona (parametry, prymitywy, kompresja, kodowanie bajtów, próbkowanie, podstawy arytmetyki wielomianów, prototyp K-PKE i KEM na mnożeniu zwykłym).

## Osoba 1: NTT i arytmetyka

- `polynomial.py`:
    - `ZETAS`, `GAMMAS`, stała `3303` (128⁻¹ mod Q)
    - `ntt` (Alg. 9), `inv_ntt` (Alg. 10)
    - `base_case_multiply` (Alg. 12), `multiply_ntts` (Alg. 11)
    - `ntt_vec`, `inv_ntt_vec` dla wektorów `(k, 256)`
    - `matvec`, `dot` oparte na `multiply_ntts`, bez zmiany sygnatur używanych w `pke.py`
- `docs/CONTRACT.md`: sekcja o `polynomial.py` (kształty, dziedzina NTT, sufiks `_hat`).
- Testy:
    - `inv_ntt(ntt(f)) == f`
    - `inv_ntt(multiply_ntts(ntt(f), ntt(g))) == schoolbook_mul(f, g)`
    - `ntt` zgodne z `R(f.tolist()).to_ntt().coeffs`
    - `ZETAS` i `GAMMAS` zgodne z kyber-py i FIPS Dodatek A
    - przypadki skrajne: zero, same `Q − 1`
- **Priorytet:** `ntt`, `inv_ntt`, `multiply_ntts`

## Osoba 2: weryfikacja i testy zgodności

- Testy wyczerpujące względem kyber-py:
    - `compress` / `decompress` dla wszystkich `x ∈ [0, 3328]` i `d ∈ {1, 4, 5, 10, 11}` oraz granice błędu z `CONTRACT.md` (832, 104, 52, 2, 1)
    - `byte_encode` / `byte_decode` w obie strony i względem `R.encode(d)` / `R.decode(bytes, d)`, w tym redukcja mod `Q` dla `d = 12`
    - `sample_ntt`: porównanie całej macierzy `(k, k, 256)` z `_generate_matrix_from_seed` (wykrywa zamianę `i, j`)
    - `sample_poly_cbd` względem `R.cbd(...)` oraz test statystyczny dla `η = 2` (rozkład 6 : 4 : 1 na 16)
- Niezależne testy NTT (mnożenie przez NTT = mnożenie zwykłe).


## Osoba 3: K-PKE i ML-KEM

- `pke.py`, przejście na NTT:
    - KeyGen: `ŝ = ntt(s)`, `ê = ntt(e)`, `t̂ = Â∘ŝ + ê`, w `ek` zakodowane `t̂`
    - Encrypt: `ŷ = ntt(y)`, `u = inv_ntt(Âᵀ∘ŷ) + e1`, `v = inv_ntt(t̂ᵀ∘ŷ) + e2 + μ` (`e1`, `e2` bez NTT)
    - Decrypt: `w = v − inv_ntt(ŝᵀ∘ntt(u))`
- `kem.py`:
    - `ml_kem_keygen_internal(params, d, z)`, `ml_kem_encaps_internal(params, ek, m)`, `ml_kem_decaps_internal(params, dk, c)`
    - `ml_kem_keygen`, `ml_kem_encaps`, `ml_kem_decaps`
    - kontrola wejść (FIPS §7.2–7.3, szczegóły do potwierdzenia): długość i moduł `ek`, długość `c` i `dk`, zgodność `H(ek)` w `dk`; błędy jako `ValueError`
    - wybór klucza w Decaps bez rozgałęzienia zależnego od wyniku porównania
- Testy:
    - z ustalonymi `d`, `z`, `m`: `k_pke_*` vs `_k_pke_*` z kyber-py, następnie `_keygen_internal`, `_encaps_internal`, `_decaps_internal`, wyniki bit w bit
    - interop w obie strony (nasz `encaps` → `decaps` kyber-py i odwrotnie)
    - KeyGen z PyCryptodome dla tego samego ziarna `d ‖ z`
    - zmieniony bajt w `c` daje `J(z ‖ c')`, zgodnie z kyber-py
    - determinizm wersji `_internal`
    - złe długości `ek` / `dk` / `c` dają `ValueError`; `ek` ze współczynnikiem `≥ Q` jest odrzucany

