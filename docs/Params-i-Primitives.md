# Parametry i funkcje skrótu: `params.py` i `crypto_primitives.py`

Ten dokument opisuje dwa pliki: [src/params.py](../src/params.py) i [src/crypto_primitives.py](../src/crypto_primitives.py). Dla każdego wyjaśnia, co w nim jest, po co to jest i dlaczego kod wygląda tak, a nie inaczej, łącznie z mniej oczywistymi elementami Pythona. Na końcu są opisane testy obu plików.

Matematyka całego ML-KEM jest w [Math-Code.md](Math-Code.md). Tutaj są szczegóły tych dwóch plików.

---

## 0. Po co są te dwa pliki

Te pliki niczego jeszcze nie szyfrują. To fundament, z którego korzystają wszystkie pozostałe moduły:

```text
params.py            liczby: Q, N, ZETA, N_INV oraz k, eta, du, dv dla każdego wariantu
crypto_primitives.py funkcje skrótu H, G, J, PRF, XOF
        │
        ├── sampling.py     XOF buduje macierz A, PRF daje bajty na szum
        ├── polynomial.py   ZETA i N_INV są potrzebne do NTT
        ├── compress.py     Q
        └── pke.py, kem.py  wszystko razem
```

Najważniejszy pomysł stojący za `crypto_primitives.py`: ML-KEM potrzebuje dużo losowych bajtów, ale chce umieć je **odtworzyć**. Przykład: macierz `A` nie jest przesyłana. W kluczu publicznym jest tylko 32-bajtowe ziarno `rho`, a obie strony wyliczają z niego tę samą macierz. Dlatego prawdziwie losowe są tylko pierwsze 32 bajty (ziarno), a wszystko inne powstaje z nich funkcjami skrótu. Funkcja skrótu dla tego samego wejścia daje zawsze to samo wyjście, ale z wyjścia nie da się odgadnąć wejścia.

---

## 1. `params.py`

### 1.1 `Q = 3329` i `N = 256`

ML-KEM liczy na wielomianach o 256 współczynnikach (`N`), a każdy współczynnik to liczba od 0 do 3328, czyli liczba modulo 3329 (`Q`). Wynik każdego działania bierze się modulo Q, czyli jako resztę z dzielenia przez 3329.

Dlaczego akurat 3329? Bo `Q − 1 = 3328 = 13 · 256`. Z tego, że 256 dzieli `Q − 1`, wynika, że modulo Q istnieje liczba, której 256. potęga daje 1 (następny punkt). Bez takiej liczby nie dałoby się zbudować NTT (Math-Code, rozdz. 5).

Zasada z [CONTRACT.md](CONTRACT.md) §1: liczba 3329 jest wpisana tylko w tym pliku, a inne pliki ją importują. Gdyby ktoś wpisał ją ręcznie w innym miejscu i się pomylił, nie dostałby żadnego błędu, tylko zły klucz, i to w zupełnie innym module.

### 1.2 `ZETA = 17`

17 to **pierwiastek pierwotny 256. stopnia z jedynki** modulo Q. Oznacza to dwie rzeczy:

1. `17²⁵⁶ mod 3329 = 1`,
2. żadna mniejsza dodatnia potęga 17 nie daje 1.

Widać to, licząc kolejne kwadraty (każda liczba to kwadrat poprzedniej, modulo 3329):

| wykładnik | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| `17^wykładnik mod Q` | 17 | 289 | 296 | 1062 | 2642 | 2580 | 1729 | **3328** | **1** |

`17¹²⁸ = 3328`, a 3328 to −1 modulo Q. Dopiero `17²⁵⁶ = (−1)² = 1`.

**Dlaczego test sprawdza tylko dwie potęgi** (`pow(17, 256, Q) == 1` i `pow(17, 128, Q) == Q − 1`)? Najmniejsza potęga, która daje 1 (tzw. rząd liczby), zawsze dzieli każdą inną potęgę dającą 1. Skoro `17²⁵⁶ = 1`, to rząd dzieli 256, czyli jest jedną z liczb 1, 2, 4, …, 256. Gdyby był mniejszy niż 256, dzieliłby też 128, a wtedy `17¹²⁸` byłoby równe 1. Tymczasem wynosi −1. Zostaje więc tylko 256.

### 1.3 `N_INV = pow(128, -1, Q)`, czyli 3303

To odwrotność liczby 128 modulo Q: taka liczba, że pomnożona przez 128 daje resztę 1 z dzielenia przez 3329. Sprawdzenie: `128 · 3303 = 422 784 = 127 · 3329 + 1`.

Jest potrzebna na samym końcu odwrotnej NTT. Odwrotna NTT ma 7 warstw, każda podwaja wartości, więc po wszystkich warstwach wynik jest pomnożony przez `2⁷ = 128`. Mnożenie przez 3303 to cofa.

Dlaczego 128, a nie 256? Pełna NTT miałaby 8 warstw i potrzebowałaby pierwiastka 512. stopnia z jedynki. Takiego modulo 3329 nie ma, bo 512 nie dzieli 3328. NTT w ML-KEM zatrzymuje się więc po 7 warstwach, na 128 parach współczynników (Math-Code, rozdz. 5).

### 1.4 Zestawy parametrów

| Zestaw | `k` | `eta1` | `eta2` | `du` | `dv` |
|---|---|---|---|---|---|
| ML-KEM-512 | 2 | 3 | 2 | 10 | 4 |
| ML-KEM-768 | 3 | 2 | 2 | 10 | 4 |
| ML-KEM-1024 | 4 | 2 | 2 | 11 | 5 |

- `k`: ile wielomianów ma wektor sekretu, a macierz `A` ma `k × k` wielomianów,
- `eta1`, `eta2`: jak duży jest szum (współczynniki od `−eta` do `eta`),
- `du`, `dv`: ile bitów zostaje z każdego współczynnika szyfrogramu po kompresji (pełny współczynnik ma 12 bitów).

Szczegóły w Math-Code, rozdz. 2.3.

### 1.5 Skąd wzory na rozmiary

Wielomian, w którym każdy z 256 współczynników jest zapisany na `d` bitach, zajmuje `256 · d / 8 = 32 · d` bajtów. Przy 12 bitach to 384 bajty. Z tego wynika wszystko:

- **`ek_size = 384·k + 32`**: wektor `t̂` to `k` wielomianów po 384 bajty, do tego ziarno `rho` (32 bajty),
- **`dk_size = 768·k + 96`**: sekret `ŝ` (`384·k`), kopia całego `ek` (`384·k + 32`), skrót `H(ek)` (32) i sekret `z` (32), razem `768·k + 96`,
- **`ct_size = 32·(du·k + dv)`**: `k` wielomianów po `du` bitów i jeden po `dv` bitów.

| Zestaw | `ek` | `dk` | szyfrogram |
|---|---|---|---|
| ML-KEM-512 | 800 | 1632 | 768 |
| ML-KEM-768 | 1184 | 2400 | 1088 |
| ML-KEM-1024 | 1568 | 3168 | 1568 |

To, że przy 1024 `ek` i szyfrogram mają tę samą długość, wynika przypadkiem z arytmetyki i nie jest regułą.

### 1.6 Elementy Pythona w tym pliku

**`@dataclass(frozen=True)`.** `@dataclass` to dekorator, który na podstawie listy pól sam tworzy konstruktor (`__init__`), czytelne wypisywanie obiektu (`__repr__`) i porównywanie (`__eq__`). `frozen=True` blokuje zmianę pól po utworzeniu obiektu:

```python
ML_KEM_768.k = 5  # dataclasses.FrozenInstanceError: cannot assign to field 'k'
```

To nie jest drobiazg. `ML_KEM_768` jest jedną wspólną stałą dla wszystkich modułów. Przypadkowe przypisanie zmieniłoby ją w całym programie, po cichu, i bardzo trudno byłoby znaleźć przyczynę. Dodatkowa korzyść: takie obiekty mogą być kluczami w słowniku.

**`@property`.** `ek_size` jest napisane jak metoda, ale odczytuje się je bez nawiasów, jak pole: `params.ek_size`. Rozmiar wynika z `k`, więc jest liczony przy odczycie, a nie zapisany osobno. Dzięki temu nie da się utworzyć zestawu z rozmiarem niezgodnym z `k`.

**`pow(128, -1, Q)`.** `pow` z trzema argumentami liczy potęgę modulo trzeci argument. Od Pythona 3.8 wykładnik `-1` oznacza odwrotność modulo. Gdy odwrotność nie istnieje, `pow` zgłasza `ValueError`. Dla zwykłych potęg jest też dużo szybszy niż `(a ** b) % Q`, bo nie liczy po drodze ogromnej liczby.

**`assert` poza funkcją.** Linia `assert 128 * N_INV % Q == 1` wykonuje się przy każdym `import params`, więc błąd zatrzymałby program od razu. Dwie uwagi:

- uruchomienie Pythona z opcją `-O` usuwa wszystkie `assert`. Dlatego nie wolno nimi sprawdzać danych od użytkownika. Do tego służy `ValueError`, jak w `prf`,
- ten konkretny `assert` w praktyce zawsze jest prawdziwy, bo `pow` właśnie policzył odwrotność. Jest bardziej czytelnym opisem niż zabezpieczeniem. Prawdziwym zabezpieczeniem jest test `N_INV == 3303`.

**`MLKEMParams("ML-KEM-512", k=2, ...)`.** Pierwszy argument (`name`) jest podany po kolei, a pozostałe po nazwie. Pięć podobnych do siebie liczb łatwo pomylić, a nazwy pól to uniemożliwiają.

---

## 2. `crypto_primitives.py`

### 2.1 Co jest w środku SHA-3

SHA-3 i SHAKE to dwie odmiany tej samej konstrukcji, nazywanej Keccak. Działa ona jak **gąbka**: ma wewnętrzny stan o rozmiarze 1600 bitów. Najpierw „wchłania” dane wejściowe porcjami, mieszając stan po każdej porcji, a potem „wyciska” z niego bajty wyniku. Wielkość jednej porcji w bajtach to `block_size` w `hashlib`:

| Funkcja | `block_size` | Długość wyniku |
|---|---|---|
| SHA3-256 | 136 | zawsze 32 B |
| SHA3-512 | 72 | zawsze 64 B |
| SHAKE128 | 168 | dowolna |
| SHAKE256 | 136 | dowolna |

Stąd liczba 840 w `sample_ntt` (i w kyber-py): `840 = 5 · 168`, czyli pięć pełnych porcji SHAKE128.

SHA3-256 i SHAKE256 mają tę samą wielkość porcji, ale **dają różne wyniki** nawet przy tej samej długości 32 bajtów. Każda z tych funkcji dokleja do wejścia inny, krótki dopisek, właśnie po to, żeby ich wyniki się różniły. Dlatego `h_hash` i `j_hash` wyglądają podobnie, ale nie są zamienne.

### 2.2 Która funkcja gdzie jest używana

| Funkcja | Co liczy | Gdzie w ML-KEM |
|---|---|---|
| `g_hash` | SHA3-512, wynik dzielony na 32 + 32 B | KeyGen: `rho, sigma = G(d + bytes([k]))`. Encaps: `K, r = G(m + H(ek))`. Decaps: `K', r' = G(m' + h)` |
| `h_hash` | SHA3-256 | skrót klucza publicznego `H(ek)`, zapisany w `dk` |
| `j_hash` | SHAKE256, 32 B | klucz zastępczy przy odrzuceniu: `K̄ = J(z + c)` |
| `prf` | SHAKE256(`s + bytes([b])`), `64·eta` B | bajty dla szumu: `s`, `e`, `y`, `e1`, `e2` |
| `xof` | SHAKE128, dowolna długość | `sample_ntt`: elementy macierzy `Â` |

Cała droga przez ML-KEM:

```text
KeyGen:  d, z losowe
         rho, sigma = G(d || k)
         A   z XOF(rho || j || i)
         s, e z PRF(sigma, 0), PRF(sigma, 1), ...
         dk  = dk_pke || ek || H(ek) || z

Encaps:  m losowe
         K, r = G(m || H(ek))
         c = Encrypt(ek, m, r)                 r wyznacza cały szum szyfrowania

Decaps:  m' = Decrypt(dk_pke, c)
         K', r' = G(m' || h)
         c' = Encrypt(ek, m', r')              ponowne szyfrowanie
         jeśli c' != c: zwróć J(z || c)        klucz zastępczy zamiast błędu
```

Symbol `||` to sklejenie ciągów bajtów, w Pythonie `+` na `bytes`.

Ostatnia linia to **niejawne odrzucenie**. Atakujący, który wyśle spreparowany szyfrogram, nie dostaje komunikatu o błędzie, tylko klucz wyglądający na losowy. Nie dowiaduje się więc, czy odszyfrowanie się udało (Math-Code, rozdz. 13).

### 2.3 Funkcja po funkcji

**`h_hash`.** `hashlib.sha3_256(data)` tworzy obiekt skrótu, a `.digest()` zwraca wynik jako `bytes`. Istnieje też `.hexdigest()`, które zwraca tekst szesnastkowy, ale tu potrzebujemy bajtów.

**`j_hash`.** `hashlib.shake_256(data).digest(32)`. Długość trzeba podać, bo SHAKE z definicji nie ma domyślnej długości wyniku. Samo `.digest()` daje błąd `TypeError: digest() missing required argument 'length'`.

**`g_hash`.** 64 bajty z SHA3-512 są dzielone na dwie połowy wycinkami: `digest[:32]` (bajty 0–31) i `digest[32:]` (bajty 32–63). Adnotacja `tuple[bytes, bytes]` mówi, że funkcja zwraca parę, dzięki czemu można pisać `rho, sigma = g_hash(...)`.

Bajt `k` w `G(d + bytes([k]))` sprawia, że to samo ziarno `d` daje zupełnie inne klucze dla ML-KEM-512, 768 i 1024. Klucze różnych poziomów nie są więc ze sobą w żaden sposób powiązane. Takie celowe rozróżnianie wejść różnych zastosowań nazywa się *separacją domen*. Ten bajt był jedną ze znanych różnic między wersją roboczą a ostateczną wersją FIPS 203. Bez niego kod działałby sam ze sobą, ale nie byłby zgodny ze standardem.

**`prf(eta, s, b)`.**

1. Sprawdza argumenty: `eta` musi być 2 albo 3, `s` musi mieć 32 bajty, `b` musi mieścić się w jednym bajcie. Inaczej zgłasza `ValueError` (CONTRACT §3). Błędne wywołanie ma się wywrócić od razu tutaj, a nie objawić jako zły klucz w `pke.py`.
2. Dokleja do `s` jeden bajt `b` i liczy SHAKE256 o długości `64 · eta` bajtów.

Skąd `64 · eta`? To `512 · eta` bitów, czyli `2 · eta` bitów na każdy z 256 współczynników. Dokładnie tyle zużywa losowanie szumu (CBD). Dla `eta = 2` to 128 bajtów, dla `eta = 3` to 192.

`b` to licznik. To samo `s` z różnymi `b` daje zupełnie różne bajty, więc każdy wielomian szumu jest inny. Powtórzenie `b` oznaczałoby dwa identyczne szumy, a to poważnie osłabiłoby bezpieczeństwo.

**`xof(seed, length)`.** FIPS 203 opisuje XOF jako strumień z trzema operacjami: rozpocznij, wchłoń dane, wyciśnij kolejne bajty. `hashlib` tak nie umie: każde `.digest(n)` zwraca `n` bajtów od samego początku. Dlatego nasza funkcja ma prostszą postać: podajesz ziarno i liczbę bajtów. Działa to dzięki temu, że dłuższy wynik SHAKE zaczyna się od krótszego:

```python
hashlib.shake_128(b"x").digest(4).hex()  # e472c5e3
hashlib.shake_128(b"x").digest(8).hex()  # e472c5e394f30ff8  (ten sam początek)
```

Jeśli `sample_ntt` zabraknie bajtów, prosi o dwa razy więcej. Pierwsze bajty są identyczne, więc wynik się nie zmienia, liczenie trwa tylko chwilę dłużej. `xof` celowo nic nie wie o indeksach `i` i `j`. Kolejność `rho || j || i` jest ustalana w jednym miejscu, w `sample_ntt` (CONTRACT §5.1).

### 2.4 Elementy Pythona w tym pliku

**`bytes([b])` a `bytes(b)`: najgroźniejsza pułapka.** Wyglądają prawie tak samo, a robią coś zupełnie innego:

```python
bytes([5])  # b'\x05'                  jeden bajt o wartości 5
bytes(5)  # b'\x00\x00\x00\x00\x00'  pięć bajtów zerowych
```

Pomyłka nie daje żadnego błędu, tylko zły wynik.

**`if eta not in {2, 3}:`** sprawdza, czy `eta` należy do zbioru `{2, 3}`. To krótszy zapis `eta != 2 and eta != 3`.

**`if not 0 <= b <= 255:`** to porównanie łańcuchowe, typowe dla Pythona. Znaczy to samo co `0 <= b and b <= 255`, a `not` obejmuje całość. W wielu innych językach taki zapis działałby inaczej.

Samo `bytes([256])` też zgłosiłoby `ValueError`, ale z mało pomocnym komunikatem `bytes must be in range(0, 256)`. Własne sprawdzenie mówi, który argument jest zły.

**`f"... {eta}"`** to tzw. f-string: wstawia wartość zmiennej do tekstu. Dzięki temu w komunikacie o błędzie widać, jaka wartość faktycznie przyszła.

---

## 3. Testy

### 3.1 Zasada: porównanie z niezależną implementacją

Nasz kod nie sprawdza sam siebie, tylko jest porównywany z kyber-py, czyli implementacją ML-KEM napisaną przez kogoś innego. Gdyby sprawdzał sam siebie, błąd mógłby być „spójny”: np. zamiana `i` z `j` przy budowie macierzy nadal pozwala odszyfrować własne szyfrogramy, więc test oparty tylko na naszym kodzie by go nie wykrył.

Uczciwe ograniczenie: kyber-py też korzysta z `hashlib`, więc testy funkcji skrótu nie sprawdzają samego SHA-3. Wykrywają natomiast błędy, które możemy popełnić my: zły algorytm, złą długość wyniku, złą kolejność sklejania. Test `xof` nie sprawdza też kolejności `j`, `i`. To zadanie testów `sample_ntt` w `test_sampling.py`.

### 3.2 `tests/test_crypto_primitives.py` (23 przypadki)

| Test | Co sprawdza |
|---|---|
| `test_hashes_match_kyber_py` | H, J i G dają to samo co kyber-py, także dla pustego i długiego wejścia |
| `test_hash_output_lengths` | wyniki mają po 32 bajty |
| `test_prf_matches_kyber_py` | PRF zgodny z kyber-py dla obu `eta` i kilku `b`, w tym skrajnych 0 i 255 |
| `test_xof_matches_kyber_py` | `xof` zgodny z kyber-py |
| `test_xof_longer_output_extends_shorter_one` | dłuższy wynik zaczyna się od krótszego (na tym opiera się `sample_ntt`) |
| `test_g_with_k_domain_separation_gives_rho_of_ek` | `G(d ‖ k)` daje `rho` równe końcówce prawdziwego klucza publicznego, a `G(d)` nie |
| `test_prf_rejects_invalid_arguments` | złe argumenty `prf` dają `ValueError` |
| `test_xof_rejects_negative_length` | ujemna długość daje `ValueError` |

Druga część testu `G` to tzw. kontrola negatywna. Sprawdza, że test nie przeszedłby, gdyby bajt `k` nie miał znaczenia. Dobry test musi umieć się nie udać.

### 3.3 `tests/test_params.py` (2 testy)

- `test_ring_constants`: `17²⁵⁶ = 1`, `17¹²⁸ = −1`, `N_INV == 3303` i `128 · N_INV mod Q == 1`,
- `test_parameter_sets_match_kyber_py_and_encoded_sizes`: parametry każdego zestawu zgadzają się z kyber-py, a wyliczone `ek_size`, `dk_size` i `ct_size` są równe długościom prawdziwych kluczy i szyfrogramu wygenerowanych przez kyber-py.

### 3.4 Elementy Pythona i pytest w testach

**`@pytest.mark.parametrize("data", [...])`** uruchamia ten sam test dla każdej wartości z listy i raportuje każdy przypadek osobno. Gdy coś padnie, od razu widać, dla jakich danych.

**Dwa `parametrize` jeden nad drugim** dają wszystkie kombinacje. Jeśli jeden ma 2 wartości, a drugi 3, test uruchomi się 6 razy. Dlatego w raporcie jest 23 przypadki, choć funkcji testowych jest 8.

**Parametryzacja kilkoma nazwami naraz:** `("eta", "s", "b")` z listą trójek rozpakowuje każdą trójkę na trzy argumenty testu.

**`with pytest.raises(ValueError):`** test przechodzi tylko wtedy, gdy kod w bloku zgłosi ten wyjątek. Jeśli funkcja zwróci wynik bez błędu, test pada.

**`_H`, `_prf`, `_k_pke_keygen` z kyber-py:** podkreślenie na początku nazwy oznacza w Pythonie metodę wewnętrzną, która może się zmienić w nowej wersji biblioteki. Korzystamy z nich świadomie (CONTRACT §8), dlatego kyber-py jest przypięte do wersji `1.2.0` w `requirements.txt` i `pyproject.toml`.

**`from kyber_py.ml_kem import ML_KEM_768 as KYBER_ML_KEM_768`:** alias (`as …`) jest konieczny, bo nazwa pokrywa się z naszą stałą. To różne obiekty: nasz to tylko zestaw liczb, a obiekt z kyber-py umie generować klucze.

**`from params import Q` bez `src.`** działa dzięki ustawieniu `pythonpath = ["src"]` w `pyproject.toml`. Bez pytesta (albo bez tego ustawienia) taki import by się nie udał.

**`ML_KEM(DEFAULT_PARAMETERS[kyber_name])`** tworzy obiekt kyber-py ze słownika parametrów. `params.name.replace("ML-KEM-", "ML")` zamienia naszą nazwę na klucz używany w kyber-py, np. `"ML-KEM-768"` na `"ML768"`.

**`bytes(range(32))`** daje 32 bajty o wartościach 0, 1, …, 31. To ustalone ziarno, więc test przy każdym uruchomieniu robi dokładnie to samo.

**Kolejność importów w `test_params.py`** (`ML_KEM_512, …, N_INV, ZETA, MLKEMParams, N, Q`) wygląda na przypadkową, ale to automatyczne sortowanie narzędzia ruff. Najpierw idą nazwy uznane za stałe (same wielkie litery, więcej niż jeden znak), potem nazwy uznane za klasy. Jednoliterowe `N` i `Q` ruff traktuje jak klasy.

**`tests/test_primitives.py`** sprawdza tylko, czy środowisko działa (NumPy, `hashlib`, kyber-py). Używa `int32`, choć CONTRACT §2.1 ustala `int64` dla wielomianów. W tym teście to bez znaczenia, ale nie należy przenosić tego do kodu.

---

## 4. Ściąga: miejsca, które wyglądają inaczej, niż działają

| Miejsce | Wygląda jak | Naprawdę |
|---|---|---|
| `pow(128, -1, Q)` | potęga ujemna | odwrotność modulo Q |
| `params.ek_size` | pole | metoda liczona przy każdym odczycie |
| `frozen=True` | drobna opcja | ochrona wspólnej stałej przed zmianą |
| `assert` poza funkcją | sprawdzanie danych | znika po `python -O` |
| `bytes([b])` | to samo co `bytes(b)` | jeden bajt, a nie `b` bajtów zerowych |
| `.digest(32)` w SHAKE | argument opcjonalny | argument obowiązkowy |
| `G(d + bytes([k]))` | zbędny bajt | wymagany przez ostateczny FIPS 203 |
| `xof(seed, length)` | uproszczenie | działa, bo dłuższy SHAKE zaczyna się od krótszego |
| `0 <= b <= 255` | dwa porównania | jedno wyrażenie łańcuchowe |
| dwa `parametrize` | dwa przebiegi | wszystkie kombinacje wartości |
