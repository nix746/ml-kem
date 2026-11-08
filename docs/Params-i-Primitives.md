# Krok 1 i 2: parametry i funkcje skrótu

Opis tego, co robią [src/params.py](../src/params.py) i [src/crypto_primitives.py](../src/crypto_primitives.py), po co to jest i dlaczego kod wygląda tak, jak wygląda. Uzupełnia [Math-Code.md](Math-Code.md): tam jest matematyka całego ML-KEM, tu są szczegóły tych dwóch plików oraz ich testów.

W kodzie są też tymczasowe komentarze z dopiskiem `[TMP]`. Odsyłają do sekcji tego dokumentu i **nie powinny trafić do commita** (usuwanie: [sekcja 5](#5-usuwanie-komentarzy-tmp)).

---

## 0. Obraz całości

Te dwa pliki to fundament, który nic jeszcze nie szyfruje. Wszystkie pozostałe moduły z nich korzystają:

```text
params.py            -> liczby: Q, N, ZETA, N_INV, k, eta, du, dv
crypto_primitives.py -> losowość deterministyczna: H, G, J, PRF, XOF
        |
        +-- sampling.py     XOF -> macierz A,  PRF -> szum s, e
        +-- polynomial.py   ZETA, N_INV -> NTT
        +-- compression.py  Q, du, dv
        +-- pke.py / kem.py wszystko razem
```

Kluczowy pomysł: ML-KEM potrzebuje dużo losowych bajtów, ale chce je **odtwarzać**. Macierz $A$ nie jest przesyłana, tylko 32-bajtowe ziarno $\rho$, z którego obie strony wyliczają identyczną macierz. Dlatego prawie cała „losowość” po pierwszych 32 bajtach to funkcje skrótu z ziarna: to samo wejście daje zawsze to samo wyjście.

---

## 1. `params.py`

### 1.1 `Q = 3329` i `N = 256`

Pracujemy w pierścieniu $R_q = \mathbb{Z}_{3329}[X]/(X^{256}+1)$, czyli wielomiany mają 256 współczynników modulo 3329. Wartość 3329 wybrano dlatego, że $q - 1 = 3328 = 13 \cdot 2^8$, a to gwarantuje istnienie pierwiastka 256. stopnia z jedynki, którego wymaga NTT (Math-Code §1.1).

Zasada z CONTRACT §1: liczba 3329 jest wpisana tylko tutaj. Jeśli inny moduł wpisze ją na sztywno, literówka nie wyjdzie jako błąd składni, tylko jako zły klucz trzy moduły dalej.

### 1.2 `ZETA = 17`

To pierwiastek pierwotny 256. stopnia z jedynki modulo $q$. Tak wygląda ciąg kolejnych kwadratów (policzone w Pythonie):

| $e$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| $17^{2^e} \bmod q$ | 17 | 289 | 296 | 1062 | 2642 | 2580 | 1729 | **3328** | **1** |

Dlaczego test `pow(17, 256, Q) == 1` i `pow(17, 128, Q) == Q - 1` wystarcza, choć sprawdza tylko dwie potęgi? Z pierwszej równości rząd liczby 17 dzieli 256, więc jest potęgą dwójki. Gdyby był mniejszy niż 256, dzieliłby 128 i wtedy $17^{128}$ byłoby równe 1. Tymczasem wynosi $-1$, więc rząd to dokładnie 256. Ta liczba jest potrzebna, żeby rozłożyć $X^{256}+1$ w NTT.

### 1.3 `N_INV = pow(128, -1, Q)`, czyli 3303

Odwrotność 128 modulo $q$: $128 \cdot 3303 = 422784 = 127 \cdot 3329 + 1$.

Dlaczego 128, a nie 256? Zwykła NTT wymagałaby pierwiastka 512. stopnia, a takiego modulo 3329 nie ma ($512 \nmid 3328$). Transformata zatrzymuje się więc po 7 warstwach na 128 parach współczynników, a odwrotna NTT dzieli przez $2^7 = 128$ (Math-Code §4.4).

### 1.4 Zestawy parametrów

| Zestaw | $k$ | $\eta_1$ | $\eta_2$ | $d_u$ | $d_v$ |
|---|---|---|---|---|---|
| ML-KEM-512 | 2 | 3 | 2 | 10 | 4 |
| ML-KEM-768 | 3 | 2 | 2 | 10 | 4 |
| ML-KEM-1024 | 4 | 2 | 2 | 11 | 5 |

Znaczenie każdego parametru opisuje Math-Code §1.3. W skrócie: $k$ to rozmiar macierzy, $\eta$ to szerokość szumu, a $d_u, d_v$ to liczba bitów zostawianych po kompresji szyfrogramu.

### 1.5 Skąd wzory na rozmiary

Liczba zakodowana na $d$ bitach dla 256 współczynników zajmuje $256 \cdot d / 8 = 32d$ bajtów. Z tego wynika wszystko:

- **`ek_size = 384k + 32`:** wektor $\hat t$ to $k$ wielomianów po 12 bitów ($32 \cdot 12 = 384$ B każdy) plus ziarno $\rho$ (32 B).
- **`dk_size = 768k + 96`:** tajny $\hat s$ ($384k$), cały `ek` ($384k + 32$), `H(ek)` (32) i `z` (32). Razem $768k + 96$.
- **`ct_size = 32(du·k + dv)`:** $k$ wielomianów po $d_u$ bitów plus jeden po $d_v$ bitów.

| Zestaw | `ek` | `dk` | `ct` |
|---|---|---|---|
| ML-KEM-512 | 800 | 1632 | 768 |
| ML-KEM-768 | 1184 | 2400 | 1088 |
| ML-KEM-1024 | 1568 | 3168 | 1568 |

Równość `ek` i `ct` przy 1024 to przypadek arytmetyki, nie zasada.

### 1.6 Python w tym pliku

**`@dataclass(frozen=True)`.** `dataclass` sam generuje `__init__`, `__repr__` i `__eq__` z listy pól. `frozen=True` blokuje zmianę po utworzeniu:

```python
ML_KEM_768.k = 5   # dataclasses.FrozenInstanceError: cannot assign to field 'k'
```

To nie kosmetyka. `ML_KEM_768` jest globalną stałą współdzieloną przez wszystkie moduły, więc przypadkowe przypisanie zepsułoby ją w całym programie, po cichu i trudno do wyśledzenia. Efekt uboczny: obiekty są haszowalne, więc mogą być kluczem słownika.

**`@property`.** `ek_size` jest metodą, ale wywołuje się ją bez nawiasów: `params.ek_size`. Rozmiar wynika z `k`, więc jest liczony, a nie przechowywany. Dzięki temu nie da się utworzyć zestawu z niespójnym rozmiarem. W `frozen` dataclass zwykłe pola też dałyby się ustawić tylko raz, ale wtedy trzeba by je podawać ręcznie.

**`pow(128, -1, Q)`.** Trzyargumentowy `pow` liczy potęgę modularnie, a od Pythona 3.8 akceptuje wykładnik `-1`, co oznacza odwrotność modularną. Jeśli odwrotność nie istnieje, rzuca `ValueError`. Dla zwykłych potęg jest też o rzędy wielkości szybszy niż `(a ** b) % Q`, bo nie buduje olbrzymiej liczby pośredniej.

**`assert` na poziomie modułu.** Wykonuje się przy każdym `import params`, więc błąd zatrzymałby program od razu. Dwie uwagi:
- `python -O` usuwa wszystkie `assert`, dlatego w kodzie produkcyjnym nie wolno nimi walidować danych wejściowych. Do tego służy `ValueError` w `prf`.
- Ten assert jest w praktyce zawsze prawdziwy, skoro `pow` policzył właśnie odwrotność. Pełni rolę czytelnej dokumentacji i wymogu z instrukcji. Prawdziwe zabezpieczenie to test `N_INV == 3303`.

**`MLKEMParams("ML-KEM-512", k=2, ...)`.** Mieszanie argumentów pozycyjnych i nazwanych: `name` podany po kolei, reszta po nazwie, żeby nie pomylić pięciu podobnych liczb.

---

## 2. `crypto_primitives.py`

### 2.1 Co jest w środku: gąbka Keccak

SHA-3 i SHAKE to ten sam Keccak, działający jak gąbka na stanie 1600 bitów. Wejście jest wchłaniane porcjami, potem stan jest mieszany i wyciskane są bajty wyjścia. Parametr `block_size` z `hashlib` to wielkość tej porcji w bajtach:

| Funkcja | `block_size` | Długość wyjścia |
|---|---|---|
| SHA3-256 | 136 | stałe 32 B |
| SHA3-512 | 72 | stałe 64 B |
| SHAKE128 | 168 | dowolna |
| SHAKE256 | 136 | dowolna |

Stąd wartość 168 w testach i 840 ($5 \cdot 168$) w kyber-py: to pełne bloki SHAKE128.

SHA3-256 i SHAKE256 mają taki sam blok, ale **nie dają tego samego wyniku** dla 32 bajtów, bo każda funkcja dokleja do wejścia inny dopisek. Dlatego `h_hash` i `j_hash` wyglądają podobnie, a nie są zamienne.

### 2.2 Która funkcja gdzie jest użyta

| Funkcja | Definicja | Gdzie w ML-KEM |
|---|---|---|
| `g_hash` | SHA3-512, podział 32 + 32 | KeyGen: `rho, sigma = G(d + bytes([k]))`. Encaps: `K, r = G(m + H(ek))`. Decaps: `K', r' = G(m' + h)` |
| `h_hash` | SHA3-256 | Skrót klucza: `H(ek)`, zapisany w `dk` |
| `j_hash` | SHAKE256, 32 B | Niejawne odrzucenie: `K_bar = J(z + c)` |
| `prf` | SHAKE256(s \|\| b), $64\eta$ B | Bajty dla CBD: szum `s`, `e`, `y`, `e1`, `e2` |
| `xof` | SHAKE128, dowolna długość | `sample_ntt`: elementy macierzy $\hat A$ |

Pełny przepływ w KEM:

```text
KeyGen:  d, z losowe
         rho, sigma = G(d || k)
         A   z XOF(rho || j || i)
         s,e z PRF(sigma, 0), PRF(sigma, 1), ...
         dk  = dk_pke || ek || H(ek) || z

Encaps:  m losowe
         K, r = G(m || H(ek))
         c = Encrypt(ek, m, r)                 # r steruje całym szumem

Decaps:  m' = Decrypt(dk_pke, c)
         K', r' = G(m' || h)
         c' = Encrypt(ek, m', r')              # ponowne szyfrowanie
         jeśli c' != c: zwróć J(z || c)        # fałszywy klucz zamiast błędu
```

Ostatnia linia to niejawne odrzucenie. Atakujący wysyłający spreparowany szyfrogram nie dostaje komunikatu o błędzie, tylko pseudolosowy klucz i nie dowiaduje się, czy deszyfrowanie się udało.

### 2.3 Szczegóły funkcji

**`h_hash`:** `hashlib.sha3_256(data)` tworzy obiekt, a `.digest()` zwraca wynik jako `bytes`. Istnieje też `.hexdigest()` zwracający tekst, ale tu potrzebujemy bajtów.

**`j_hash`:** `shake_256(data).digest(32)`. Długość jest obowiązkowa, bo SHAKE z definicji nie ma domyślnej: `digest_size` wynosi u niego 0, a samo `.digest()` daje `TypeError: digest() missing required argument 'length'`.

**`g_hash`:** 64 bajty dzielone wycinkami `[:32]` i `[32:]`. Adnotacja `tuple[bytes, bytes]` mówi, że zwracana jest para, dzięki czemu można pisać `rho, sigma = g_hash(...)`.

Bajt `k` w `G(d || k)` to separacja domen: to samo ziarno `d` daje zupełnie inne klucze dla ML-KEM-512, 768 i 1024, więc klucze różnych poziomów nie są ze sobą powiązane. Pominięcie tego bajtu było znaną różnicą między draftem a finalnym FIPS 203. Taki błąd daje implementację, która działa sama ze sobą, ale jest niezgodna ze standardem.

**`prf(eta, s, b)`:**
- Rozmiar $64\eta$ bajtów to $512\eta$ bitów, czyli $2\eta$ bitów na każdy z 256 współczynników. Tyle zużywa CBD. Dla $\eta = 2$ to 128 B, dla $\eta = 3$ to 192 B.
- `b` to licznik: to samo `s` i różne `b` dają niezależne strumienie, stąd każdy wielomian szumu ma inne bajty. Powtórzenie `b` oznaczałoby powtórzenie szumu, a to jest katastrofa dla bezpieczeństwa.
- Walidacja rzuca `ValueError` (CONTRACT §3), bo pomyłka powinna wybuchnąć tutaj, a nie jako zły klucz w `pke.py`.

**`xof(seed, length)`:** najciekawsza z tych funkcji, bo FIPS 203 opisuje XOF jako strumień (`Init`, `Absorb`, `Squeeze`), a `hashlib` nie ma trybu strumieniowego: każde `.digest(n)` zwraca `n` bajtów od początku. Dlatego kontrakt jest prosty: podajesz ziarno i długość. Działa to dzięki własności prefiksu:

```python
shake_128(b"x").digest(4).hex()  # e472c5e3
shake_128(b"x").digest(8).hex()  # e472c5e394f30ff8  <- ten sam początek
```

Jeśli `sample_ntt` zabraknie bajtów, prosi o dwa razy więcej i wynik się nie zmienia, tylko kosztuje chwilę więcej. Funkcja celowo nie wie nic o `i` i `j`: kolejność `rho || j || i` jest w jednym miejscu, w `sample_ntt` (CONTRACT §5.1).

### 2.4 Python w tym pliku

**`bytes([b])` kontra `bytes(b)`: najgroźniejsza pułapka.** Wyglądają prawie tak samo, a robią coś innego:

```python
bytes([5])  # b'\x05'                 jeden bajt o wartości 5
bytes(5)    # b'\x00\x00\x00\x00\x00' pięć bajtów zerowych
```

Pomyłka nie daje żadnego błędu, tylko zły wynik.

**`if eta not in {2, 3}:`** to test przynależności do zbioru `{...}`. Równoważne `eta != 2 and eta != 3`, tylko krótsze.

**`if not 0 <= b <= 255:`** to porównanie łańcuchowe, typowe dla Pythona: znaczy `0 <= b and b <= 255`, a `not` obejmuje całość. W wielu innych językach to by nie zadziałało tak samo.

Samo `bytes([256])` też rzuciłoby `ValueError`, ale z mało mówiącym komunikatem `bytes must be in range(0, 256)`. Własny test podaje, który argument jest zły.

**`f"... {eta}"`** to f-string, czyli wstawienie wartości do tekstu. Dzięki temu w komunikacie widać, co faktycznie przyszło.

**`s + bytes([b])`:** `+` na `bytes` skleja. To jest operator `||` z FIPS 203.

---

## 3. Testy

### 3.1 Skąd wiadomo, że to działa

Podstawowa zasada: **wyrocznia zamiast samosprawdzania**. Własny kod nie sprawdza sam siebie, tylko jest porównywany z kyber-py, czyli niezależną implementacją. Bez tego bug mógłby być spójny sam ze sobą: np. zamiana `i` i `j` nadal pozwala odszyfrować własne szyfrogramy.

Uczciwe ograniczenie: kyber-py też używa `hashlib`, więc testy hashy nie sprawdzają samego SHA-3. Wyłapują natomiast to, co można zepsuć u nas: zły algorytm, złą długość, złą kolejność sklejania. Test `xof` też nie sprawdza kolejności `j`, `i`: to zadanie testu dla `sample_ntt`.

### 3.2 Co sprawdza `test_crypto_primitives.py`

| Test | Co dowodzi |
|---|---|
| `test_hashes_match_kyber_py` | H, J, G zgodne z wyrocznią, także dla pustego i długiego wejścia |
| `test_hash_output_lengths` | Wyjścia mają po 32 B |
| `test_prf_matches_kyber_py` | PRF zgodny dla wszystkich kombinacji $\eta$ i `b`, także skrajnych 0 i 255 |
| `test_xof_matches_kyber_py` | Wynik `xof` taki sam jak `_xof` |
| `test_xof_longer_output_extends_shorter_one` | Własność prefiksu, na której polega `sample_ntt` |
| `test_g_with_k_domain_separation_gives_rho_of_ek` | `G(d \|\| k)` daje $\rho$ z prawdziwego `ek_pke`, a `G(d)` nie |
| `test_prf_rejects_invalid_arguments` | Złe argumenty dają `ValueError` |
| `test_xof_rejects_negative_length` | Jak wyżej dla długości |

Druga asercja w teście `G` to kontrola negatywna: upewnia się, że test nie przeszedłby, gdyby bajt `k` nie miał znaczenia. Dobry test musi umieć się nie udać.

### 3.3 Python i pytest

**`@pytest.mark.parametrize("data", [...])`** uruchamia ten sam test dla każdej wartości i raportuje je osobno, więc wiadomo, który przypadek padł.

**Dwa `parametrize` jeden na drugim** dają iloczyn kartezjański. W `test_prf_matches_kyber_py` to $2 \cdot 3 = 6$ przypadków. Stąd 28 testów w raporcie, choć funkcji testowych jest mniej.

**Parametryzacja wieloma nazwami:** `("eta", "s", "b")` z listą trójek rozpakowuje każdą trójkę na trzy argumenty.

**`with pytest.raises(ValueError):`** test przechodzi tylko wtedy, gdy blok rzuci ten wyjątek. Jeśli funkcja zwróci wynik, test pada.

**`_H`, `_prf`, `_k_pke_keygen`:** podkreślenie oznacza metodę prywatną, która może się zmienić między wersjami. Świadomy kompromis z CONTRACT §7: dlatego kyber-py ma być przypięty do `1.2.0`. Na tej gałęzi `pyproject.toml` ma jeszcze `>=1.0.0`, przypięcie jest na `contract-and-attributes`.

**`from kyber_py.ml_kem import ML_KEM_768 as KYBER_ML_KEM_768`:** alias jest konieczny, bo nazwa pokrywa się z naszą stałą. Co ważne, to różne obiekty: nasz jest samą listą parametrów, a ten z kyber-py umie generować klucze.

**`from params import Q` bez `src.`** działa dzięki `pythonpath = ["src"]` w `pyproject.toml`. Bez pytesta i bez tego ustawienia import by się nie udał.

**`ML_KEM(DEFAULT_PARAMETERS[kyber_name])`** tworzy obiekt kyber-py ze słownika parametrów. `params.name.replace("ML-KEM-", "ML")` zamienia naszą nazwę na klucz kyber-py, np. `ML768`.

**`bytes(range(32))`** daje bajty 0, 1, ..., 31. To ustalone ziarno, dzięki czemu test zachowuje się tak samo przy każdym uruchomieniu.

**Dziwna kolejność importów w `test_params.py`** (`N_INV, ZETA, MLKEMParams, N, Q`): to nie pomyłka, tylko sortowanie ruffa. Najpierw stałe, potem klasy. Jednoliterowe `N` i `Q` są traktowane jak klasy, bo stałą jest dla niego nazwa z samych wielkich liter dłuższa niż jeden znak.

**`test_primitives.py`** sprawdza tylko środowisko (NumPy, hashlib, kyber-py). Używa `int32`, a CONTRACT §2.1 ustala `int64` dla wielomianów. W tym teście nie ma to znaczenia, ale nie należy tego przenosić do `polynomial.py`.

---

## 4. Ściąga: najdziwniejsze miejsca

| Miejsce | Wygląda jak | Naprawdę |
|---|---|---|
| `pow(128, -1, Q)` | Potęga ujemna | Odwrotność modularna |
| `params.ek_size` | Pole | Metoda liczona przy każdym odczycie |
| `frozen=True` | Drobna opcja | Ochrona globalnej stałej przed zmianą |
| `assert` w module | Walidacja | Znika po `python -O` |
| `bytes([b])` | To samo co `bytes(b)` | Jeden bajt, a nie `b` zer |
| `.digest(32)` | Argument opcjonalny | W SHAKE obowiązkowy |
| `G(d + bytes([k]))` | Nadmiarowy bajt | Separacja domen wymagana przez FIPS 203 |
| `xof(seed, length)` | Uproszczenie | Działa dzięki własności prefiksu SHAKE |
| `0 <= b <= 255` | Dwa porównania | Jedno wyrażenie łańcuchowe |
| Dwa `parametrize` | Dwa przebiegi | Iloczyn kartezjański |

---

## 5. Usuwanie komentarzy `[TMP]`

Każdy tymczasowy komentarz zajmuje całą linię i zawiera `[TMP]`, więc usunięcie tych linii przywraca kod dokładnie do stanu bez komentarzy. Przed commitem uruchom w **Git Bash**, w katalogu projektu:

```bash
git grep --untracked -l "\[TMP\]" -- src tests | xargs -d '\n' sed -b -i '/\[TMP\]/d'
git grep --untracked -n "\[TMP\]" -- src tests
```

Drugie polecenie nie powinno niczego wypisać. Flaga `-b` jest potrzebna: bez niej `sed` w Git Bash zamienia końce wierszy CRLF na LF, a `tests/test_params.py` ma CRLF.
