# Wielomiany i NTT: `polynomial.py`

Ten dokument opisuje cały plik [src/polynomial.py](../src/polynomial.py) i jego testy: [tests/test_polynomial.py](../tests/test_polynomial.py), [tests/test_ntt.py](../tests/test_ntt.py) i [tests/test_ntt_independent.py](../tests/test_ntt_independent.py). Matematyka jest w [Math-Code.md](Math-Code.md), rozdziały 3–6, a parametry i funkcje skrótu w [Params-i-Primitives.md](Params-i-Primitives.md).

Plik ma trzy części:

| Część | Funkcje | Do czego |
|---|---|---|
| 1. Zwykła arytmetyka | `poly_add`, `poly_sub`, `schoolbook_mul` | dodawanie i odejmowanie w całym ML-KEM, mnożenie szkolne jako wzorzec w testach |
| 2. NTT | `ZETAS`, `GAMMAS`, `ntt`, `inv_ntt`, `base_case_multiply`, `multiply_ntts` | mnożenie wielomianów w postaci, której wymaga standard |
| 3. Wektory i macierze | `ntt_vec`, `inv_ntt_vec`, `transpose`, `dot`, `matvec` | działania typu `A·s` i `tᵀ·y` w K-PKE |

---

## 1. Po co ten plik

Cały ML-KEM to działania na wielomianach: `t = A·s + e`, `u = Aᵀ·y + e1`, `v = tᵀ·y + e2 + μ`. Ten plik daje do nich wszystkie funkcje.

Mnożenie ma tu dwie wersje i warto wiedzieć, która do czego służy:

- **`schoolbook_mul`** to mnożenie „jak w szkole”. Jest proste i łatwe do sprawdzenia ręcznie, dlatego służy jako **wzorzec w testach NTT**. Kod schematu (`pke.py`) z niego nie korzysta.
- **`multiply_ntts`** mnoży wielomiany w postaci NTT. Tej wersji (przez `matvec` i `dot`) używa `pke.py`, bo standard zapisuje klucze w postaci NTT.

Najważniejszy test NTT to `inv_ntt(multiply_ntts(ntt(f), ntt(g))) == schoolbook_mul(f, g)`. NTT jest trudne i łatwo w nim o subtelny błąd, a mnożenie szkolne jest proste. Dlatego w tym pliku czytelność kodu jest ważniejsza niż szybkość.

---

## 2. Zwykła arytmetyka

### 2.1 Wielomian jako tablica

Wielomian `f = f₀ + f₁·X + … + f₂₅₅·X²⁵⁵` to tablica NumPy o kształcie `(256,)` i typie `int64`. `f[i]` to współczynnik przy `X^i`. Współczynniki są liczbami od 0 do 3328 (CONTRACT §2.1).

Przykład: `5 + 2·X²` to tablica `[5, 0, 2, 0, …, 0]` z 256 elementami.

Dodawanie i odejmowanie działają współczynnik po współczynniku, jak zwykłe dodawanie wielomianów, tylko z wynikiem modulo 3329:

```python
return (a + b) % Q  # poly_add
return (a - b) % Q  # poly_sub
```

### 2.2 Funkcja pomocnicza `_as_int64_pair`

Wołają ją `poly_add`, `poly_sub`, `schoolbook_mul` i `multiply_ntts`. Zamienia oba argumenty na tablice `int64`, sprawdza, że mają ten sam kształt i że ostatni wymiar ma długość 256. W przeciwnym razie zgłasza `ValueError`.

Ponieważ sprawdzany jest tylko **ostatni** wymiar, `poly_add` i `poly_sub` działają też na wektorach wielomianów o kształcie `(k, 256)`. Osobna funkcja `vec_add` nie jest potrzebna.

### 2.3 Zawijanie ze zmianą znaku

ML-KEM liczy modulo wielomian `X²⁵⁶ + 1`, czyli obowiązuje reguła `X²⁵⁶ = −1`. Każdą potęgę od 256 w górę można zmniejszyć:

```text
X^(256 + k) = X^256 · X^k = −X^k
```

Pełny iloczyn dwóch wielomianów ma stopień do 510, więc dzieli się go na dwie części:

```text
c = [c_0 … c_255 | c_256 … c_511]
     zostaje       każde c_(256+k) odejmuje się od c_k
```

Stąd wzór `(c[:256] − c[256:]) % Q`. Przykłady z testów:

- `X · X²⁵⁵ = X²⁵⁶ = −1`, czyli wynik `[3328, 0, 0, …]`,
- `X¹⁰⁰ · X²⁰⁰ = X³⁰⁰ = −X⁴⁴`, czyli 3328 na pozycji 44.

Takie mnożenie nazywa się **negacyklicznym**. Przy zwykłym zawijaniu (`X²⁵⁶ = 1`) byłby tam plus. Ten minus to najważniejszy znak w całej funkcji: zamiana go na plus psuje 45 testów (punkt 6.4).

### 2.4 `schoolbook_mul` krok po kroku

```python
c = np.zeros(2 * N, dtype=np.int64)
for i in range(N):
    c[i : i + N] += a[i] * b
return (c[:N] - c[N:]) % Q
```

Mnożymy `a · b = Σ a_i · X^i · b`. Dla każdego `i`:

1. `a[i] * b` mnoży cały wielomian `b` przez jedną liczbę,
2. mnożenie przez `X^i` przesuwa wynik o `i` pozycji w prawo, stąd wycinek `c[i : i + N]`,
3. `+=` dodaje to do wyniku.

Mały przykład dla `N = 3`, `a = 1 + 2X`, `b = 3 + 4X`:

```text
i = 0: c[0:3] += 1 · [3, 4, 0]   →  c = [3,  4, 0, 0, 0, 0]
i = 1: c[1:4] += 2 · [3, 4, 0]   →  c = [3, 10, 8, 0, 0, 0]
i = 2: c[2:5] += 0 · [3, 4, 0]   →  bez zmian
```

Wynik to `3 + 10X + 8X²`, tak samo jak przy mnożeniu ręcznym.

**Dlaczego jedna pętla, skoro mnożenie jest podwójne?** Wewnętrzną pętlę (mnożenie całej tablicy `b` przez liczbę i dodanie) wykonuje NumPy w języku C. W Pythonie jest tylko 256 przebiegów zamiast 65 536. Pomiar: ok. 0,8 ms na jedno mnożenie, a `np.convolve` ok. 0,05 ms.

**Dlaczego tablica ma 512 elementów, skoro iloczyn ma 511 współczynników?** Wtedy `c[:256]` i `c[256:]` mają po 256 elementów i można je od siebie odjąć. Ostatni element `c[511]` zawsze zostaje zerem.

### 2.5 Wycinki, widoki i kopie

**`c[i : i + N]` to widok, a nie kopia.** Wycinek tablicy NumPy wskazuje na te same dane, więc `+=` na wycinku zmienia `c`:

```python
c = np.zeros(4, dtype=np.int64)
v = c[1:3]
v += 5
c  # [0, 5, 5, 0]
```

W zwykłych listach Pythona jest inaczej: wycinek listy to kopia. Stąd też pułapka w teście „funkcja nie zmienia wejścia”: samo `a_before = a` nic by nie dało, bo to ta sama tablica pod inną nazwą. Dlatego test robi `a.copy()`.

**`np.asarray` nie kopiuje, gdy typ już się zgadza.** Dla tablicy `int64` zwraca dokładnie ten sam obiekt, a dla `int32` tworzy nową tablicę. Dlatego żadna funkcja nie zmienia `a` ani `b` w miejscu: `a + b` zawsze tworzy nową tablicę, a `a += b` zniszczyłoby dane wywołującego.

**`a.shape[-1:]`** to wycinek krotki z kształtem, zawierający tylko ostatni wymiar: dla `(3, 256)` daje `(256,)`, a dla pojedynczej liczby (kształt `()`) daje `()`. Zwykłe `a.shape[-1]` dla pojedynczej liczby zgłosiłoby `IndexError` zamiast obiecanego `ValueError`.

**Własne sprawdzanie kształtów** jest potrzebne, bo bez niego NumPy po cichu „rozciągnąłby” tablicę `(1,)` do `(256,)` i dodał jedną liczbę do każdego współczynnika. Takie rozciąganie nazywa się *broadcasting*. W `a[i] * b` jest przydatne, ale przy złych danych ukrywa błąd.

### 2.6 `%` z liczbami ujemnymi

| Wyrażenie | Wynik |
|---|---|
| `-1 % 3329` w Pythonie i NumPy | `3328` |
| `np.fmod(-1, 3329)` | `-1` |
| `-1 % 3329` w C | `-1` |

Python i NumPy dają wynik `%` ze znakiem dzielnika, więc przy dodatnim Q wynik jest zawsze od 0 do Q − 1. Dzięki temu `(a − b) % Q` nie wymaga poprawiania ujemnych różnic.

### 2.7 Dlaczego `int64`

Do współczynnika przy `X²⁵⁵` wchodzi 256 par (wszystkie `i + j = 255`), więc w najgorszym razie suma to:

```text
256 · 3328² = 2 835 349 504  >  2 147 483 647 = 2³¹ − 1 (największy int32)
```

Tablice NumPy nie zgłaszają przepełnienia, tylko zawijają wynik. `np.convolve` na dwóch tablicach `int32` z samymi 3328 zwraca największą wartość 2 137 587 712 zamiast 2 835 349 504, bez żadnego ostrzeżenia. Wynik wygląda jak normalna liczba, więc błąd wychodzi dopiero jako zły iloczyn.

Pojedynczy iloczyn to tylko ok. 11 milionów i mieści się w `int32`. **Najważniejsze jest więc `int64` dla tablicy `c`, w której sumują się iloczyny.** Sprawdziliśmy to, psując kod celowo: `c` w `int32` psuje 5 testów (punkt 6.4). Zamiana wejść na `int64` w `_as_int64_pair` jest po to, żeby wynik zawsze był `int64`, zgodnie z CONTRACT.

---

## 3. NTT

### 3.1 Co liczy NTT

Wielomian `X²⁵⁶ + 1` rozkłada się na iloczyn 128 wielomianów stopnia 2:

```text
X²⁵⁶ + 1 = (X² − γ₀)(X² − γ₁) … (X² − γ₁₂₇),   γᵢ = 17^(2·BitRev₇(i) + 1) mod Q
```

**NTT wielomianu `f` to 128 reszt z dzielenia `f` przez kolejne czynniki `X² − γᵢ`.** Reszta z dzielenia przez wielomian stopnia 2 ma postać `a + b·X`, czyli dwie liczby. 128 reszt po dwie liczby to znowu 256 liczb. Para numer `i` leży na pozycjach `2i` i `2i + 1`.

W reszcie z dzielenia przez `X² − γ` obowiązuje `X² = γ`. Stąd proste przykłady, które nasze `ntt` daje dokładnie tak:

| `f` | reszta z dzielenia przez `X² − γ` | `ntt(f)` |
|---|---|---|
| `1` | `1`, czyli para `(1, 0)` | `[1, 0, 1, 0, 1, 0, …]` |
| `X` | `X`, czyli para `(0, 1)` | `[0, 1, 0, 1, 0, 1, …]` |
| `X²` | `γ`, czyli para `(γ, 0)` | `[17, 0, 3312, 0, 2761, 0, …]` (kolejne `GAMMAS`) |

Mnożenie jest w tej postaci łatwe, bo reszta iloczynu to iloczyn reszt. Żeby pomnożyć dwa wielomiany, wystarczy pomnożyć ich pary o tych samych numerach, każdą osobno (punkt 4).

Dlaczego pary, a nie pojedyncze liczby? Rozkład dalej, na 256 czynników stopnia 1, wymagałby pierwiastka 512. stopnia z jedynki modulo 3329, a takiego nie ma (512 nie dzieli 3328).

**Po co to w ML-KEM?** Standard zapisuje klucz publiczny i prywatny w postaci NTT. Bez NTT klucze nie byłyby zgodne z żadną inną implementacją. NTT jest też zwykle szybsze, ale w naszym kodzie nie: `ntt` i `inv_ntt` to pętle Pythona po pojedynczych liczbach. Pomiar: pełne mnożenie przez NTT (`ntt`, `ntt`, `multiply_ntts`, `inv_ntt`) trwa ok. 2 ms, a `schoolbook_mul` ok. 0,8 ms. Kod ma być jak najprostszy, więc tego nie optymalizujemy.

### 3.2 Tablice `ZETAS` i `GAMMAS`

```python
ZETAS = [pow(ZETA, _bitrev7(i), Q) for i in range(128)]
GAMMAS = [pow(ZETA, 2 * _bitrev7(i) + 1, Q) for i in range(128)]
```

`_bitrev7(i)` zapisuje `i` na 7 bitach i czyta od tyłu: `format(i, "07b")` daje napis z siedmioma cyframi binarnymi, `[::-1]` odwraca napis, a `int(…, 2)` czyta go z powrotem jako liczbę.

| `i` | binarnie | od tyłu | `_bitrev7(i)` |
|---|---|---|---|
| 1 | `0000001` | `1000000` | 64 |
| 2 | `0000010` | `0100000` | 32 |
| 3 | `0000011` | `1100000` | 96 |
| 64 | `1000000` | `0000001` | 1 |

Odwrócona kolejność bitów sprawia, że algorytm NTT bierze kolejne mnożniki z tablicy po prostu po kolei (`k = 1, 2, 3, …`).

Pierwsze wartości, zgodne z FIPS 203, dodatek A:

- `ZETAS`: `1, 1729, 2580, 3289, 2642, 630, 1897, 848, …` (mnożniki w `ntt` i `inv_ntt`),
- `GAMMAS`: `17, 3312, 2761, 568, 583, 2746, 2649, 680, …` (liczby `γᵢ` z rozkładu, używane w `multiply_ntts`).

`GAMMAS` idą parami `γ, −γ`: `17 + 3312 = 3329`, `2761 + 568 = 3329`. Przy każdym rozbiciu `X⁴ − c = (X² − √c)(X² + √c)` powstają dwa czynniki z przeciwnym znakiem.

### 3.3 `_as_poly`

Kopiuje wejście do nowej tablicy `int64` i sprawdza, że to jeden wielomian `(256,)`. Używa `np.array`, które **zawsze kopiuje**, w odróżnieniu od `np.asarray`. Kopia jest konieczna, bo `ntt` i `inv_ntt` zmieniają tablicę w miejscu. Bez niej zniszczyłyby tablicę wywołującego.

### 3.4 `ntt` (Alg. 9)

```python
k = 1
length = 128
while length >= 2:
    for start in range(0, N, 2 * length):
        zeta = ZETAS[k]
        k += 1
        for j in range(start, start + length):
            t = (zeta * f[j + length]) % Q
            f[j + length] = (f[j] - t) % Q
            f[j] = (f[j] + t) % Q
    length //= 2
```

1. `length` przyjmuje wartości 128, 64, 32, 16, 8, 4, 2. Każda to jedna **warstwa**, razem 7.
2. W warstwie tablica jest podzielona na **bloki** długości `2 · length`: w pierwszej warstwie 1 blok, w drugiej 2, w trzeciej 4 i tak dalej.
3. Każdy blok dostaje swój mnożnik `ZETAS[k]`. `k` rośnie od 1 do 127, bo bloków jest razem `1 + 2 + … + 64 = 127`. `ZETAS[0] = 1` nie jest używane.
4. W bloku element `j` z pierwszej połowy łączy się z elementem `j + length` z drugiej połowy przez **motylek**: `t = zeta · f[j+length]`, potem `f[j+length] = f[j] − t` i `f[j] = f[j] + t`.

Co robi motylek? Blok to reszta z dzielenia przez pewien wielomian `X^(2·length) − c`, który rozkłada się na `(X^length − z)(X^length + z)`, gdzie `z = zeta`. Dzielimy blok na dolną połowę `lo` i górną `hi`, czyli `f = lo + X^length · hi`:

- reszta z dzielenia przez `X^length − z`: wstawiamy `X^length = z` i wychodzi `lo + z·hi`, czyli nowe `f[j]`,
- reszta z dzielenia przez `X^length + z`: wstawiamy `X^length = −z` i wychodzi `lo − z·hi`, czyli nowe `f[j + length]`.

Każda warstwa rozbija każdy blok na dwa mniejsze. Po 7 warstwach bloki mają długość 2, czyli są gotowymi parami.

### 3.5 `inv_ntt` (Alg. 10)

To ta sama procedura puszczona od końca: warstwy w odwrotnej kolejności (`length = 2, 4, …, 128`), `k` maleje od 127 do 1, a motylek jest odwrotny:

```python
t = f[j]
f[j] = (t + f[j + length]) % Q
f[j + length] = (zeta * (f[j + length] - t)) % Q
```

Z `a = lo + z·hi` i `b = lo − z·hi` odzyskuje połówki. Suma `a + b = 2·lo` trafia na dolną pozycję. Na górną trafia `zeta · (b − a) = zeta · (−2·z·hi)`. Mnożnik wzięty od końca tablicy spełnia `zeta · z = −1`, więc wychodzi `2·hi`.

Każda warstwa odzyskuje połówki pomnożone przez 2, więc po 7 warstwach wszystko jest pomnożone przez `2⁷ = 128`. Na końcu kod mnoży przez `N_INV = 3303`, czyli odwrotność 128 modulo Q, i to cofa.

---

## 4. Mnożenie w postaci NTT

### 4.1 `base_case_multiply` (Alg. 12)

Mnoży dwie pary, czyli wielomiany `a0 + a1·X` i `b0 + b1·X`, z regułą `X² = γ`:

```text
(a0 + a1·X)(b0 + b1·X) = a0·b0 + (a0·b1 + a1·b0)·X + a1·b1·X²
                       = (a0·b0 + a1·b1·γ) + (a0·b1 + a1·b0)·X
```

Stąd w kodzie `c0 = (a0*b0 + a1*b1*gamma) % Q` i `c1 = (a0*b1 + a1*b0) % Q`. Przykład: `(1 + 2X)(3 + 4X)` przy `γ = 17` daje `(139, 10)`.

### 4.2 `multiply_ntts` (Alg. 11)

Dla każdego `i` od 0 do 127 mnoży parę `(f_hat[2i], f_hat[2i+1])` przez parę `(g_hat[2i], g_hat[2i+1])` funkcją `base_case_multiply` z `γ = GAMMAS[i]` i zapisuje wynik na tych samych pozycjach.

Każda para ma **swoje** `γ`. Użycie jednej wartości dla wszystkich par daje wynik, który wygląda normalnie, ale jest błędny (psuje 30 testów, punkt 6.4).

`int(…)` przy argumentach zamienia liczby NumPy na zwykłe liczby Pythona, które nie mają limitu wielkości. Iloczyn `a1 * b1 * gamma` może dojść do ok. 37 miliardów, więc na pewno się nie przepełni.

Pełne mnożenie przez NTT to więc:

```python
inv_ntt(multiply_ntts(ntt(f), ntt(g)))  # daje to samo co schoolbook_mul(f, g)
```

Dodawanie działa tak samo w obu postaciach (NTT jest liniowe), więc wyniki w postaci NTT można dodawać zwykłym `poly_add`.

---

## 5. Wektory i macierze

| Funkcja | Wejście | Wyjście | Co robi |
|---|---|---|---|
| `ntt_vec(v)` | `(k, 256)` | `(k, 256)` | `ntt` na każdym wielomianie |
| `inv_ntt_vec(v_hat)` | `(k, 256)` | `(k, 256)` | `inv_ntt` na każdym wielomianie |
| `transpose(a_hat)` | `(k, k, 256)` | `(k, k, 256)` | `wynik[i, j] = a_hat[j, i]` |
| `dot(u_hat, v_hat)` | 2 × `(k, 256)` | `(256,)` | `u[0]·v[0] + u[1]·v[1] + …` |
| `matvec(a_hat, v_hat)` | `(k, k, 256)`, `(k, 256)` | `(k, 256)` | element `i` to `dot(a_hat[i], v_hat)` |

Przykład dla `k = 2`:

```text
matvec(A, s) = [ A[0,0]·s[0] + A[0,1]·s[1] ,
                 A[1,0]·s[0] + A[1,1]·s[1] ]
```

`dot` i `matvec` mnożą przez `multiply_ntts`, więc **oba argumenty muszą już być w postaci NTT**, a wynik też jest w postaci NTT. Podanie zwykłych wielomianów nie zgłosi błędu, tylko da bezsensowny wynik. Dlatego takie zmienne mają w nazwie przyrostek `_hat` (CONTRACT §2.3).

`transpose` zamienia tylko dwa pierwsze wymiary: `np.transpose(a_hat, (1, 0, 2))`. Samo `a_hat.T` odwróciłoby wszystkie trzy wymiary i przemieszałoby też współczynniki wielomianów. `.copy()` robi z wyniku osobną tablicę. Iloczyn `Aᵀ · y` z szyfrowania liczy się jako `matvec(transpose(a_hat), y_hat)`.

`*` i `@` z NumPy **nie mnożą wielomianów**. `a * b` mnoży współczynnik po współczynniku, co w zwykłej postaci jest błędne. Trzeba użyć `schoolbook_mul`, `multiply_ntts`, `dot` albo `matvec`.

Tak wyglądają obliczenia w `pke.py` (od PR #9):

```python
s_hat = ntt_vec(s)
t_hat = poly_add(matvec(a_hat, s_hat), ntt_vec(e))  # KeyGen
y_hat = ntt_vec(y)
u = poly_add(inv_ntt_vec(matvec(transpose(a_hat), y_hat)), e1)  # Encrypt
v = poly_add(poly_add(inv_ntt(dot(t_hat, y_hat)), e2), mu)  # Encrypt
w = poly_sub(v, inv_ntt(dot(s_hat, ntt_vec(u))))  # Decrypt
```

---

## 6. Testy

### 6.1 `tests/test_polynomial.py` (50 przypadków)

Zwykła arytmetyka jest sprawdzana trzema niezależnymi sposobami:

| Sposób | Co wykrywa | Ograniczenie |
|---|---|---|
| `convolve_mul` (pełny iloczyn przez `np.convolve` i to samo zawinięcie) | błędy w pętli i indeksach | ma ten sam wzór `c[:N] − c[N:]`, więc **nie wykryje błędu w zawijaniu** |
| `GenericPolynomialRing` z kyber-py | także błędy zawijania, bo to zupełnie osobna implementacja | mało przypadków, bo jest wolna |
| przypadki policzone ręcznie | konkretne własności matematyczne | pokrywają tylko wybrane wejścia |

Żaden z tych sposobów nie wystarcza sam. `np.convolve` jest szybki, a kyber-py zamyka lukę, której `np.convolve` nie zamknie.

**Przypadek największych współczynników.** Dla `a = [Q−1, …, Q−1]`, czyli `a = −(1 + X + … + X²⁵⁵)`, wynik `a²` da się policzyć ręcznie. Minusy się znoszą, więc `a² = (1 + X + … + X²⁵⁵)²`. W pełnym iloczynie `c_k = k + 1` dla `k ≤ 255` i `c_(256+k) = 255 − k`. Po zawinięciu `c_k − c_(256+k) = 2k − 254`. Stąd w teście oczekiwany wynik `(2 * np.arange(N) − 254) % Q`. Test działa dla wejść `int32` i `int64`, więc przy okazji sprawdza brak przepełnienia.

Inne testy: wyniki w zakresie `[0, Q)`, odejmowanie cofa dodawanie, działanie na wektorach, wejście nie jest zmieniane, wynik zawsze `int64`, `X · X²⁵⁵ = −1`, mnożenie przez 1 i przez 0, przemienność i rozdzielność, złe kształty dają `ValueError`.

### 6.2 `tests/test_ntt.py` (53 przypadki)

- `ZETAS[:8]` i `GAMMAS[:8]` równe wartościom z FIPS 203, dodatek A, całe `ZETAS` równe tablicy z kyber-py, `GAMMAS` idą parami `γ, −γ`,
- `inv_ntt(ntt(f)) == f` i `ntt(inv_ntt(f)) == f`,
- `ntt` i `inv_ntt` zgodne z `to_ntt()` i `from_ntt()` z kyber-py,
- przypadki skrajne: same 0, same 1, same `Q − 1`,
- `base_case_multiply` na przykładzie ręcznym, `multiply_ntts` równe `schoolbook_mul` i zgodne z kyber-py,
- `ntt_vec`, `inv_ntt_vec`, `transpose`, `dot` i `matvec` dla `k = 2, 3, 4`, porównane z sumami `schoolbook_mul`,
- wejście nie jest zmieniane, złe kształty i wektory różnej długości dają `ValueError`.

Funkcje pomocnicze `to_kyber`, `to_kyber_ntt`, `from_kyber`, `random_poly` i fixture `rng` są w [tests/conftest.py](../tests/conftest.py).

### 6.3 `tests/test_ntt_independent.py` (25 przypadków)

Te testy nie używają kyber-py ani niczego z `polynomial.py` poza testowanymi funkcjami. Oczekiwane wyniki liczą wprost z definicji matematycznej:

- 17 jest pierwiastkiem pierwotnym 256. stopnia z jedynki,
- iloczyn wszystkich `X² − γᵢ` naprawdę daje `X²⁵⁶ + 1`,
- `ntt(f)` daje dokładnie reszty z dzielenia `f` przez `X² − γᵢ`. Funkcja `ntt_by_definition` liczy je ze wzoru: skoro `X² = γ`, to `f mod (X² − γ) = Σ f_(2j)·γʲ + X · Σ f_(2j+1)·γʲ`,
- NTT jednomianów `X^i`, liniowość NTT,
- mnożenie przez NTT równe mnożeniu negacyklicznemu z definicji, `X · X²⁵⁵ = −1` także przez NTT,
- iloczyn małego szumu z losowym wielomianem, tak jak w K-PKE.

Dzięki temu wykryłyby błąd, który byłby wspólny dla naszego kodu i naszych pozostałych testów.

### 6.4 Czy testy naprawdę coś wykrywają

Kod zepsuto celowo w kilku miejscach (na kopii pliku) i uruchomiono wszystkie trzy pliki testów. Bez zepsucia przechodzi 128 z 128 przypadków.

| Zepsucie | Ile przypadków padło |
|---|---|
| `+` zamiast `−` przy zawijaniu w `schoolbook_mul` | 45 |
| tablica `c` w `int32` | 5 |
| brak `% Q` w `poly_sub` | 4 |
| brak mnożenia przez `N_INV` w `inv_ntt` | 49 |
| `k` nie maleje w `inv_ntt` | 51 |
| jedna `γ` dla wszystkich par w `multiply_ntts` | 30 |

### 6.5 Elementy pytest

**`parametrize` mnoży przypadki.** W `test_polynomial.py` jest 13 funkcji testowych, ale 50 przypadków:

| Test | Parametry | Przypadki |
|---|---|---|
| `test_schoolbook_matches_convolve` | 20 ziaren | 20 |
| `test_operations_reject_wrong_shapes` | 3 operacje × 3 pary kształtów | 9 |
| `test_operations_do_not_modify_input_and_return_int64` | 3 operacje × 2 typy | 6 |
| `test_schoolbook_matches_kyber_py` | 5 ziaren | 5 |

**Parametrem może być funkcja.** `[poly_add, poly_sub, schoolbook_mul]` to lista funkcji, a `operation(a, b)` w teście wywołuje kolejną z nich.

**`np.random.default_rng(seed)`** z ustalonym ziarnem daje te same „losowe” dane przy każdym uruchomieniu, więc błąd zawsze da się odtworzyć (CONTRACT §8).

**`.tolist()`** zamienia liczby `numpy.int64` na zwykłe liczby Pythona, które nie mają limitu wielkości. kyber-py liczy na takich liczbach, więc przy okazji chroni to wzorzec przed przepełnieniem.

**`np.array_equal(x, y)`.** Samo `x == y` na tablicach daje tablicę wartości `True`/`False`, a nie jedną wartość, więc `assert x == y` zgłosiłoby błąd.

---

## 7. Ściąga: miejsca, które wyglądają inaczej, niż działają

| Miejsce | Wygląda jak | Naprawdę |
|---|---|---|
| `c[i : i + N] += ...` | zmiana kopii | zapis do `c` przez widok |
| `(c[:N] - c[N:])` | zwykłe odejmowanie | reguła `X²⁵⁶ = −1` |
| `np.zeros(2 * N)` | o jeden za dużo | dwie równe połowy, `c[511]` zawsze 0 |
| `dtype=np.int64` dla `c` | ostrożność | warunek poprawnego wyniku |
| `(a - b) % Q` | może dać liczbę ujemną | zawsze od 0 do Q − 1 |
| `np.asarray(a, ...)` | kopia | ten sam obiekt, gdy typ się zgadza |
| `np.array(f, ...)` w `_as_poly` | to samo co `asarray` | zawsze kopia, bo `ntt` zmienia tablicę w miejscu |
| `a.shape[-1:]` | `a.shape[-1]` | krotka, bez `IndexError` dla pojedynczej liczby |
| pętla z jednym `for` w `schoolbook_mul` | mnożenie liniowe | druga pętla jest w NumPy (w C) |
| `k = 1` na początku `ntt` | pominięty element | `ZETAS[0] = 1` nie jest potrzebne |
| `N_INV` = odwrotność 128 | powinno być 256 | NTT ma 7 warstw, a nie 8 |
| `ntt(X²)` | coś skomplikowanego | po prostu kolejne `GAMMAS` |
| `a_hat.T` | transpozycja macierzy | odwraca też współczynniki; trzeba `transpose` |
