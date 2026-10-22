
> Stałe: $q = 3329$, $n = 256$, $k \in {2,3,4}$ (ML-KEM-512 / 768 / 1024). `ndarray` = tablica numpy.

## 1. **Parametry** - FIPS, kyber-py 

### 1.1 Skąd te liczby

- q = 3329 = 13 * 256 + 1, czyli 256 | q-1 istnieje pierwiastek 256-go stopnia z jedynki: $\zeta = 17$.
- Dzięki temu da się zrobić NTT (szybkie mnożenie wielomianów).

### 1.2 Zestawy parametrów

|Wariant|$k$|$\eta_1$|$\eta_2$|$d_u$|$d_v$|
|---|---|---|---|---|---|
|512|2|3|2|10|4|
|768|3|2|2|10|4|
|1024|4|2|2|11|5|

- **W kodzie:** zwykłe stałe + `dataclass` / `NamedTuple` na zestaw.

#### 1.3 Co znaczą parametry

- $k$: liczba wielomianów w wektorze (s, e, y) i rozmiar macierzy A (k × k). Im większe $k$, tym wyższy poziom bezpieczeństwa i większe klucze.
- $\eta_1$: rozpiętość szumu CBD dla s, e (w KeyGen) i y (w Encrypt). Współczynniki są z $[-\eta_1, \eta_1]$.
- $\eta_2$: to samo dla e_1 i e_2 w Encrypt.
- $d_u$: na ilu bitach kompresujemy każdy współczynnik wektora u (zamiast 12).
- $d_v$: to samo dla wielomianu v.