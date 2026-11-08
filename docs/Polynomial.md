# Krok 3: arytmetyka w pierścieniu $R_q$

Opis podstawowej części [src/polynomial.py](../src/polynomial.py): `poly_add`, `poly_sub` i `schoolbook_mul` oraz ich testów w [tests/test_polynomial.py](../tests/test_polynomial.py). Matematyka całości jest w [Math-Code.md](Math-Code.md) §2, a parametry i funkcje skrótu w [Params-i-Primitives.md](Params-i-Primitives.md).

Tymczasowe komentarze `[TMP]` w tych plikach odsyłają do sekcji tego dokumentu. Polecenie do ich usunięcia: [Params-i-Primitives.md, sekcja 5](Params-i-Primitives.md#5-usuwanie-komentarzy-tmp).

---

## 1. Po co to jest

Cały ML-KEM to działania na wielomianach: `t = A·s + e`, `u = Aᵀ·r + e1`, `v = tᵀ·r + e2 + m`. Dodawanie i odejmowanie z tego pliku trafią wprost do `pke.py`. Mnożenie jest inne:

| Funkcja | Rola |
|---|---|
| `poly_add`, `poly_sub` | Prawdziwe operacje używane w szyfrowaniu i deszyfrowaniu |
| `schoolbook_mul` | **Wyrocznia do testów NTT**, nie trafi na ścieżkę KEM |
| `multiply_ntts` (kolejny krok) | Właściwe, szybkie mnożenie używane w KEM |

Późniejszy test NTT będzie sprawdzał `inv_ntt(multiply_ntts(ntt(f), ntt(g))) == schoolbook_mul(f, g)` (Math-Code §4.3). NTT jest trudna i łatwo o subtelny błąd, a `schoolbook_mul` jest prosta i sprawdzalna ręcznie. Dlatego czytelność jest tu ważniejsza niż szybkość.

---

## 2. Matematyka

### 2.1 Wielomian jako tablica

$f = f_0 + f_1 X + \dots + f_{255} X^{255}$ to tablica o kształcie `(256,)`, gdzie `f[i]` to współczynnik przy $X^i$. Współczynniki są modulo $Q = 3329$ w zakresie `[0, Q)` (CONTRACT §2.1).

Dodawanie i odejmowanie działają po współczynnikach, jak zwykłe dodawanie wielomianów, tylko modulo $Q$.

### 2.2 Zawijanie: dlaczego minus

Pracujemy modulo $X^{256} + 1$, więc $X^{256} = -1$. Każdą potęgę od 256 w górę można zredukować:

$$X^{256+k} = X^{256} \cdot X^k = -X^k$$

Pełny iloczyn ma stopień do 510, więc dzieli się na dwie części:

```text
c = [c_0 ... c_255 | c_256 ... c_511]
     zostaje          każdy c_(256+k) odejmuje się od c_k
```

Stąd wzór `(c[:256] - c[256:]) % Q`. Przykłady z testów:

- $X \cdot X^{255} = X^{256} = -1$, czyli `Q - 1` przy $X^0$.
- $X^{100} \cdot X^{200} = X^{300} = -X^{44}$, czyli `Q - 1` przy $X^{44}$.

Taki splot ze zmianą znaku nazywa się **negacyklicznym**. W zwykłym splocie cyklicznym ($X^{256} = 1$) byłby tam plus i dlatego ten minus to najważniejszy znak w całej funkcji.

### 2.3 Dlaczego „schoolbook”

To mnożenie jak w szkole, każdy wyraz przez każdy, czyli $256^2 = 65\,536$ iloczynów. NTT robi to samo w $O(n \log n)$, więc w KEM używamy jej. Tutaj złożoność nie ma znaczenia, bo funkcja działa tylko w testach.

---

## 3. Kod

### 3.1 `schoolbook_mul` krok po kroku

```python
c = np.zeros(2 * N, dtype=np.int64)
for i in range(N):
    c[i : i + N] += a[i] * b
return (c[:N] - c[N:]) % Q
```

Mnożymy $a \cdot b = \sum_i a_i X^i \cdot b$. Dla każdego `i`:

- `a[i] * b` mnoży cały wielomian `b` przez jedną liczbę,
- przemnożenie przez $X^i$ przesuwa go o `i` pozycji, stąd wycinek `c[i : i + N]`,
- `+=` dodaje to do wyniku.

Mały przykład dla `N = 3`, $a = 1 + 2X$, $b = 3 + 4X$:

```text
i=0: c[0:3] += 1 * [3, 4, 0]  ->  c = [3,  4, 0, 0, 0, 0]
i=1: c[1:4] += 2 * [3, 4, 0]  ->  c = [3, 10, 8, 0, 0, 0]
i=2: c[2:5] += 0 * [3, 4, 0]  ->  bez zmian
```

Wynik $3 + 10X + 8X^2$, zgodny z ręcznym wymnożeniem.

**Dlaczego to jedna pętla, skoro mnożenie jest podwójne?** Wewnętrzną pętlę wykonuje NumPy w C. W Pythonie jest tylko 256 przebiegów, a nie 65 536. Dlatego pomiar daje 0,33 ms na mnożenie, a `np.convolve` 0,024 ms.

**Dlaczego długość 512, skoro współczynników jest 511?** Wtedy `c[:256]` i `c[256:]` mają po 256 elementów i można je odjąć. Ostatni element `c[511]` zawsze zostaje zerem.

### 3.2 Wycinki, widoki i kopie

**`c[i : i + N]` to widok, nie kopia.** Wycinek w NumPy wskazuje na te same dane, więc `+=` zapisuje do `c`:

```python
c = np.zeros(4, dtype=np.int64)
v = c[1:3]
v += 5
c  # [0, 5, 5, 0]
```

To inaczej niż w listach Pythona, gdzie wycinek kopiuje. Stąd też pułapka w teście „nie modyfikuje wejścia”: samo `a_before = a` nic nie daje, bo to ta sama tablica, dlatego test używa `a.copy()`.

**`np.asarray` nie kopiuje, gdy typ się zgadza.** Dla tablicy `int64` zwraca dokładnie ten sam obiekt. Dla `int32` tworzy nową tablicę. Nie modyfikujemy więc `a` ani `b` w miejscu: `(a + b)` zawsze tworzy nową tablicę, a `a += b` zniszczyłoby dane wywołującego.

**`a.shape[-1:]`** to wycinek krotki z ostatnim wymiarem: `(3, 256)` daje `(256,)`, a skalar `()` daje `()`. Zwykłe `a.shape[-1]` dla skalara rzuciłoby `IndexError` zamiast obiecanego `ValueError`.

**Własna kontrola kształtów** jest potrzebna, bo bez niej NumPy po cichu rozciągnąłby `(1,)` do `(256,)` i dodał liczbę do każdego współczynnika. Takie rozciąganie to broadcasting. Przydaje się w `a[i] * b`, ale przy złych danych ukrywa błąd.

### 3.3 `%` z liczbami ujemnymi

| Wyrażenie | Wynik |
|---|---|
| `-1 % 3329` w Pythonie i NumPy | `3328` |
| `np.fmod(-1, 3329)` | `-1` |
| `-1 % 3329` w C | `-1` |

Python i NumPy biorą znak dzielnika, więc wynik jest zawsze w `[0, Q)`. Dzięki temu `(a - b) % Q` nie wymaga poprawiania ujemnych różnic.

### 3.4 Dlaczego `int64`

Do współczynnika przy $X^{255}$ wchodzi 256 par ($i + j = 255$), więc w najgorszym razie suma to:

$$256 \cdot 3328^2 = 2\,835\,349\,504 > 2\,147\,483\,647 = 2^{31} - 1$$

Tablice NumPy nie zgłaszają przepełnienia, tylko zawijają wynik: `np.convolve` na `int32` zwraca 2 137 587 712 bez żadnego ostrzeżenia. Wynik wygląda jak liczba, więc błąd wychodzi dopiero jako zły wynik mnożenia.

Pojedynczy iloczyn to tylko około $1{,}1 \cdot 10^7$ i mieści się w `int32`. **Kluczowe jest więc `int64` dla tablicy `c`, gdzie gromadzą się sumy.** Test mutacyjny to potwierdził: `c` w `int32` psuje testy, a rzutowanie tylko jednego wejścia niczego nie zmienia, bo `int32 * int64` i tak daje `int64`. Rzutowanie wejścia w `_as_int64_pair` jest po to, żeby wynik zawsze był `int64` zgodnie z kontraktem.

Uwaga: [project_notes.md](project_notes.md) podaje jeszcze `int32`. Obowiązuje `int64`.

---

## 4. Testy

### 4.1 Trzy niezależne sprawdzenia

| Sprawdzenie | Co wykrywa | Ograniczenie |
|---|---|---|
| `convolve_mul` | Błędy w pętli i indeksach | Ma ten sam wzór `c[:N] - c[N:]`, więc **nie wykryje błędu w zawijaniu** |
| `GenericPolynomialRing` z kyber-py | Także błędy zawijania, bo to osobna implementacja na listach | Mało przypadków, bo jest wolna |
| Przypadki ręczne | Konkretne własności matematyczne | Pokrywają tylko wybrane wejścia |

Żadne z nich nie wystarcza samo. `np.convolve` pełni rolę szybkiego wzorca z instrukcji, a kyber-py zamyka lukę, której wzorzec sam nie zamknie.

### 4.2 Przypadek największych współczynników

Dla `a = [Q-1, ..., Q-1]`, czyli $a = -(1 + X + \dots + X^{255})$, wynik można policzyć ręcznie, co daje oczekiwaną wartość niezależną od kodu.

$a^2 = (1 + X + \dots + X^{255})^2$, bo minusy się znoszą. W zwykłym iloczynie:

- $c_k = k + 1$ dla $k \le 255$,
- $c_{256+k} = 255 - k$.

Po zawinięciu: $c_k - c_{256+k} = (k + 1) - (255 - k) = 2k - 254$. Stąd w teście `(2 * np.arange(N) - 254) % Q`.

Test działa dla `int32` i `int64`, więc przy okazji sprawdza brak przepełnienia.

### 4.3 Czy testy naprawdę coś łapią

Kod zepsuto celowo w kilku miejscach:

| Zepsucie | Ile testów padło |
|---|---|
| `+` zamiast `-` przy zawijaniu | 28 |
| `c` w `int32` | 3 |
| Brak `% Q` w `poly_sub` | 3 |
| Brak rzutowania obu wejść na `int64` | 2 |

Ostatni wiersz powstał z dopisania testu: wcześniej nic nie sprawdzało, że wejście `int32` daje wynik `int64`.

### 4.4 Składnia pytest

**Liczba testów:** 50 w tym pliku, choć funkcji jest 13. `parametrize` mnoży przypadki:

| Test | Parametry | Przypadki |
|---|---|---|
| `test_schoolbook_matches_convolve` | 20 ziaren | 20 |
| `test_operations_reject_wrong_shapes` | 3 operacje · 3 kształty | 9 |
| `test_operations_do_not_modify_input...` | 3 operacje · 2 typy | 6 |
| `test_schoolbook_matches_kyber_py` | 5 ziaren | 5 |

**Parametrem może być funkcja:** `[poly_add, poly_sub, schoolbook_mul]` to lista funkcji, a `operation(a, b)` wywołuje kolejną z nich.

**`np.random.default_rng(seed)`:** ustalone ziarno daje te same „losowe” dane przy każdym uruchomieniu, więc błąd da się odtworzyć (CONTRACT §7).

**`.tolist()`** zamienia `numpy.int64` na zwykłe `int` Pythona, które nie mają limitu zakresu. kyber-py operuje na takich liczbach, więc to przy okazji chroni wyrocznię przed przepełnieniem.

**`np.array_equal(x, y)`:** samo `x == y` na tablicach daje tablicę prawd i fałszów, a nie jedną wartość, więc `assert x == y` rzuciłoby błąd.

---

## 5. Ściąga: najdziwniejsze miejsca

| Miejsce | Wygląda jak | Naprawdę |
|---|---|---|
| `c[i : i + N] += ...` | Zmiana kopii | Zapis do `c` przez widok |
| `(c[:N] - c[N:])` | Zwykłe odejmowanie | $X^{256} = -1$ |
| `np.zeros(2 * N)` | Jeden za dużo | Dwie równe połówki, `c[511]` zawsze 0 |
| `dtype=np.int64` dla `c` | Ostrożność | Warunek poprawności wyniku |
| `(a - b) % Q` | Może dać wynik ujemny | Zawsze w `[0, Q)` |
| `np.asarray(a, ...)` | Kopia | Ten sam obiekt, gdy typ się zgadza |
| `a.shape[-1:]` | `a.shape[-1]` | Krotka, bez `IndexError` dla skalara |
| `a_before = a.copy()` | Nadmiarowe | Bez tego test nic nie sprawdza |
| Pętla z jednym `for` | Mnożenie liniowe | Druga pętla jest w C |
