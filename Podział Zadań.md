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


# Faza 3: wizualizacje i raport

**Cel:** zestaw wykresów i raport
**Termin:** 30.12 
**Założenie:** Wizualizacje tylko importują funkcje z `src/`, a jeśli trzeba coś podmienić (np. skalę szumu), kopiują funkcję do skryptu, tak jak robi to `demo.py`.

## Przebieg

- Krok po kroku pokazuje istniejące `demo.py --pause` w terminalu. Wykresy robimy w matplotlib, każdy jako osobny skrypt w `viz/`, który zapisuje PNG do `viz/out/`. Te same pliki trafiają do raportu i na slajdy lub przy odpalaniu z terminala jakoś je wyświetlić w odpowiednim momencie
- Najważniejsze są punkty 5 i 6: pokazują, że deszyfrowanie działa tylko dzięki temu, że szum jest mały, i co się dzieje, gdy przestaje być mały. Reszta wykresów prowadzi do tego momentu.
- Okrąg `Z_q` (punkty 4–6) to jedna wspólna funkcja: kąt `2π·x/q` na wykresie 
- Wykresy są jednocześnie eksperymentami do sekcji 4 raportu (Experiments, 30 pkt), więc każdy wykres ma mieć krótki opis: co mierzymy, jakie parametry, co z tego wynika.

## Lista wizualizacji

1. **Schemat wymiany klucza Alice ↔ Bob:** KeyGen → `ek` → Encaps → `c` → Decaps → `K`, przy strzałkach rozmiary w bajtach dla 512 / 768 / 1024 (`ek` 800 / 1184 / 1568, `dk` 1632 / 2400 / 3168, `c` 768 / 1088 / 1568, `K` 32). Prosty, ale bez niego reszta nie ma kontekstu. Otwiera raport i prezentację.
2. **Próbkowanie CBD:** histogram współczynników z `sample_poly_cbd` dla `η = 2` i `η = 3` na tle rozkładu teoretycznego. Pokazuje, że szum jest mały i skupiony wokół zera.
3. **Kompresja:** wykres schodkowy `x → decompress(compress(x))` dla `d = 1, 4, 10, 11` i wykres błędu zaokrąglenia z granicami. Pokazuje, że kompresja dokłada kolejny mały błąd.
4. **Kodowanie wiadomości na okręgu `Z_q`:** bit 0 → 0, bit 1 → ⌈q/2⌉, zaznaczone obszary decyzji (odległość q/4).
5. **Dlaczego deszyfrowanie działa:** prawdziwe współczynniki `w = v − ŝᵀu` na tym samym okręgu (i jako histogram). Widać szum wokół 0 i q/2, wszystkie punkty mieszczą się w marginesie q/4.
6. **Zwiększanie szumu aż do błędów:** mnożymy `e`, `e1`, `e2` przez 1, 2, 4, 8… Okręgi dla kilku czynników, na których punkty wychodzą poza margines, i krzywa „czynnik szumu → % błędnych bitów `m`”. Wymaga kopii `k_pke_encrypt` / `k_pke_keygen` z parametrem skali w skrypcie.
7. **Problem LWE na zabawkowym przykładzie:** 1D, małe `q` (np. 97), jeden sekret `s`. Wykres punktowy par `(a, b = a·s + e mod q)`: bez szumu punkty leżą na prostych i `s` łatwo odczytać, z szumem jest chmura. Tłumaczy, skąd bierze się bezpieczeństwo.
8. **Wydajność:** czas KeyGen / Encaps / Decaps dla trzech wariantów, `time.perf_counter()`, kilkaset powtórzeń, wykres słupkowy. Porównujemy tylko nasze warianty między sobą, nie z implementacjami w C.

## Raport (po angielsku)

| Sekcja | Pkt | Co tam wchodzi | Kto |
|---|---|---|---|
| 1 Introduction | 5 | Dlaczego PQC (komputery kwantowe, „harvest now, decrypt later”), cel projektu (implementacja ML-KEM zgodna z FIPS 203 bit w bit), metodologia (implementacja w Pythonie, weryfikacja względem kyber-py, wizualizacje) | Kacper |
| 2 Background | 10 | KEM vs szyfrowanie, LWE → Module-LWE, pierścień `Z_q[X]/(X²⁵⁶+1)`, NTT, K-PKE i transformacja Fujisakiego–Okamoto (implicit rejection), parametry 512 / 768 / 1024. Tu trafiają wykresy 1, 4 i 7 | Kacper |
| 3 Testbed Setup (Implementation) | 30 | Środowisko (Python, numpy, pytest, ruff, `kyber-py==1.2.0`), struktura repo, podział na moduły, dwie fazy (prototyp na mnożeniu zwykłym → NTT), `CONTRACT.md`. Każdy opisuje moduły, które pisał w fazach 1–2 | wstęp i środowisko: Martyna; podrozdziały: każdy swoje moduły |
| 4 Experiments (Validation) | 30 | 4.1 testy jednostkowe i zgodności z kyber-py (bit w bit, interop, implicit rejection, kontrola wejść) w tabeli; 4.2+ eksperymenty z wykresów 2, 3, 5, 6, 8 | wstęp i tabela testów: Tymon; podrozdziały: każdy swoje wykresy |
| 5 Conclusions | 10 | Wkład, czy cel osiągnięty, ograniczenia (brak stałego czasu wykonania, Python, nie do produkcji), future work (constant-time, ML-DSA, testy KAT z NIST, ACVP) | Tymon i Martyna  |
| References | 5 | FIPS 203, praca Kyber (Bos i in., 2018), Regev 2005 (LWE), Langlois–Stehlé (Module-LWE), repozytorium kyber-py, strona NIST PQC.  | Trzeba cytować chyba |
| Overall quality | 10 |  |  |

### Wytyczne formatowania (z wytycznych prowadzącego, w skrócie)

- Każdy rysunek ma podpis, każda tabela ma nagłówek **nad** tabelą. Numeracja ciągła, w tekście odwołanie po numerze („Fig. 3 shows…”, nie „the figure below”). Każdy rysunek i tabela musi być przywołana w tekście.
- Rysunki jako PNG albo wektorowo (PDF/SVG), nigdy JPEG. Rysunki z cudzych prac podpisane źródłem.
- Rozdziały i podrozdziały numerowane, tytuły wielką literą, bez kropki i dwukropka na końcu (to samo w podpisach). Abstract bez numeru. Tytuł raportu w title case.
- Każdy rozdział zaczyna się kilkoma zdaniami wstępu, dopiero potem podrozdział. Koniec rozdziału ma prowadzić do następnego.
- Akapity wyjustowane, równe wcięcie, bez akapitów jednozdaniowych i dłuższych niż 1/3 strony.
- Skrót rozwijamy przy pierwszym użyciu (KEM, LWE, NTT, CBD, FIPS…). Jednolita terminologia w całym tekście.
- Jednostki ze spacją („5 ms”, „1184 B”), ale „50%”. Separator dziesiętny to kropka. Liczby poniżej 10 słownie („three variants”).
- Cytowanie przed kropką ze spacją: „…as defined in FIPS 203 [1].”
- Jedna czcionka, jeden styl (strona bierna albo „we”, nie mieszamy). Myślnik jako en-dash (`--` w LaTeX-u).
- Na koniec: spell check, podwójne spacje, spacje przed przecinkami i po nawiasach otwierających.

## Kacper

- `viz/circle.py`: wspólna funkcja rysowania okręgu `Z_q` (najpierw, bo korzysta z niej Tymon)
- Wykres 4 (kodowanie wiadomości) i 5 (szum w deszyfrowaniu)
- Wykres 7 (LWE)
- Raport: 1 Introduction, 2 Background, podrozdziały 3 i 4 o swoich modułach i wykresach

## Martyna

- Wykres 1 (schemat Alice ↔ Bob)
- Wykres 2 (CBD) i 3 (kompresja)
- Raport: wstęp sekcji 3 i opis środowiska, podrozdziały 3 i 4 o swoich modułach i wykresach, końcowe przejście pod kątem wytycznych (Overall quality)

## Tymon

- Wykres 6 (zwiększanie szumu), oparty na `viz/circle.py`
- Wykres 8 (wydajność), jeśli zostanie w raporcie
- Raport: wstęp sekcji 4 i tabela testów, 5 Conclusions, References, podrozdziały 3 i 4 o swoich modułach i wykresach

