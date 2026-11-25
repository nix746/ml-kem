# ML-KEM od matematyki do kodu

Ten dokument tłumaczy krok po kroku, jak działa ML-KEM (standard NIST FIPS 203) i jak każdy element matematyki przekłada się na kod w `src/`. Każdy rozdział ma tę samą budowę: **co to jest i po co**, **jak to działa**, **jak wygląda w kodzie** i **na co uważać**. Na końcu rozdziału są źródła.

Rozdziały idą w kolejności, w jakiej elementy są budowane: od liczb i wielomianów, przez kodowanie i losowanie, aż do szyfrowania (K-PKE) i pełnej wymiany klucza (ML-KEM). Szczegóły Pythona w plikach `params.py`, `crypto_primitives.py` i `polynomial.py` opisują osobne dokumenty: [Params-i-Primitives.md](Params-i-Primitives.md) i [Polynomial.md](Polynomial.md). Ustalenia dotyczące typów i kształtów danych są w [CONTRACT.md](CONTRACT.md).

**Główne źródła używane w całym dokumencie:**

- [FIPS 203](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf): sam standard; numery algorytmów (Alg. 3–21) i sekcji odnoszą się do niego.
- [Naskręcki, rozdz. 41](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html): ML-KEM zaimplementowany w NumPy, z wykresami.
- [Broeckelmann](https://medium.com/@robert.broeckelmann/post-quantum-cryptography-ml-kem-the-math-7382776f6c44): intuicyjny opis matematyki bez formalizmu.
- [Lyubashevsky](https://eprint.iacr.org/2024/1287) i [Menezes](https://eprint.iacr.org/2026/1098): podstawy krat i LWE w wersji formalnej.
- [kyber-py](https://kyber-py.readthedocs.io/en/latest/kyber_py.ml_kem.html): niezależna implementacja, z którą porównujemy wyniki w testach.

**Oznaczenia:** $q = 3329$, $n = 256$, $k \in \{2, 3, 4\}$. Symbol $\|$ oznacza sklejenie ciągów bajtów (w Pythonie `+` na `bytes`). Daszek nad nazwą, np. $\hat{s}$, oznacza wielomian w postaci NTT (rozdział 5), a w kodzie odpowiada mu przyrostek `_hat`.

---

## Spis treści

0. [Obraz całości](#0-obraz-całości)
1. [Problem LWE: dlaczego to jest bezpieczne](#1-problem-lwe-dlaczego-to-jest-bezpieczne)
2. [Parametry](#2-parametry)
3. [Pierścień wielomianów $R_q$](#3-pierścień-wielomianów-r_q)
4. [Wektory i macierze wielomianów](#4-wektory-i-macierze-wielomianów)
5. [NTT: szybkie mnożenie](#5-ntt-szybkie-mnożenie)
6. [Mnożenie w postaci NTT](#6-mnożenie-w-postaci-ntt)
7. [Funkcje skrótu](#7-funkcje-skrótu)
8. [Losowanie macierzy: SampleNTT](#8-losowanie-macierzy-samplentt)
9. [Losowanie szumu: SamplePolyCBD](#9-losowanie-szumu-samplepolycbd)
10. [Kompresja](#10-kompresja)
11. [Zamiana na bajty: ByteEncode i ByteDecode](#11-zamiana-na-bajty-byteencode-i-bytedecode)
12. [K-PKE: szyfrowanie](#12-k-pke-szyfrowanie)
13. [ML-KEM: wymiana klucza i transformacja Fujisakiego-Okamoto](#13-ml-kem-wymiana-klucza-i-transformacja-fujisakiego-okamoto)
14. [Rozmiary kluczy i szyfrogramu](#14-rozmiary-kluczy-i-szyfrogramu)
15. [Ściąga: wzór, algorytm, funkcja w kodzie](#15-ściąga-wzór-algorytm-funkcja-w-kodzie)
16. [Stan implementacji i ograniczenia](#16-stan-implementacji-i-ograniczenia)

---

## 0. Obraz całości

ML-KEM to **mechanizm uzgadniania klucza** (KEM, *key encapsulation mechanism*). Dwie strony, nazwijmy je Alicja i Bob, chcą mieć wspólny tajny klucz $K$ o długości 32 bajtów, którym potem zaszyfrują rozmowę zwykłym szyfrem symetrycznym, np. AES. ML-KEM nie szyfruje samych wiadomości, tylko bezpiecznie dostarcza ten klucz.

Wymiana przebiega w trzech krokach:

```text
Alicja                                         Bob
------                                         ---
1. KeyGen()  -> klucz publiczny ek,
                klucz prywatny dk
             ---------------- ek ---------------->
                                               2. Encaps(ek) -> klucz K, szyfrogram c
             <--------------- c -----------------
3. Decaps(dk, c) -> ten sam klucz K
```

Po trzecim kroku obie strony znają $K$, a ktoś, kto podsłuchał $ek$ i $c$, nie jest w stanie go odtworzyć.

Implementacja składa się z warstw. Każda korzysta tylko z warstw pod nią:

```text
kem.py              ML-KEM: KeyGen, Encaps, Decaps (Alg. 16-21)
  |
pke.py              K-PKE: szyfrowanie 32-bajtowej wiadomości (Alg. 13-15)
  |
  +-- polynomial.py   działania na wielomianach i NTT (Alg. 9-12)
  +-- sampling.py     losowanie macierzy A i szumu (Alg. 7-8)
  +-- compress.py     kompresja i zamiana na bajty (Alg. 3-6)
  +-- crypto_primitives.py   funkcje skrótu H, G, J, PRF, XOF
  +-- params.py       liczby: q, n, k, eta, du, dv
```

Najważniejsze rozróżnienie: **K-PKE** to szyfrowanie z kluczem publicznym, które samo w sobie jest bezpieczne tylko wobec biernego podsłuchu. **ML-KEM** opakowuje je tak, żeby było bezpieczne także wtedy, gdy atakujący wysyła spreparowane szyfrogramy (rozdział 13).

---

## 1. Problem LWE: dlaczego to jest bezpieczne

### 1.1 Co to jest i po co

Bezpieczeństwo ML-KEM opiera się na problemie **LWE** (*Learning With Errors*, uczenie się z błędami). Chodzi o to, że łatwy problem staje się bardzo trudny, gdy doda się do niego mały szum.

### 1.2 Jak to działa

Weźmy układ równań liniowych modulo $q$: znamy macierz $A$ i wektor $b = A \cdot s$, szukamy wektora $s$. To jest łatwe. Eliminacja Gaussa rozwiązuje to w ułamku sekundy.

Teraz dodajmy do każdego równania mały, losowy błąd $e$:

$$b = A \cdot s + e \pmod q$$

Współczynniki $e$ są bardzo małe, np. z przedziału $[-2, 2]$, ale ich nie znamy. Eliminacja Gaussa przestaje działać, bo każdy krok rachunku powiększa błędy i po kilku krokach wynik jest bezużyteczny. Najlepsze znane metody (zarówno klasyczne, jak i na komputerach kwantowych) sprowadzają się do szukania krótkiego wektora w kracie o setkach wymiarów, co przy dobrze dobranych parametrach wymaga astronomicznej liczby operacji. To jest **problem LWE**: mając $A$ i $b$, znaleźć $s$.

Klucz publiczny ML-KEM ma właśnie taką postać: $t = A \cdot s + e$. Każdy zna $A$ i $t$, ale wyliczenie z nich tajnego $s$ to rozwiązanie LWE.

### 1.3 Wersja używana w ML-KEM: Module-LWE

Zwykłe LWE wymaga macierzy rzędu kilkuset na kilkaset liczb, co daje bardzo duże klucze. ML-KEM używa wersji **Module-LWE**: elementami macierzy i wektorów nie są pojedyncze liczby, tylko wielomiany o 256 współczynnikach (rozdział 3). Dzięki temu macierz $A$ ma tylko $k \times k$ elementów, gdzie $k$ to 2, 3 albo 4, a mimo to cała struktura odpowiada kracie o wymiarze $256 \cdot k$, czyli 512, 768 lub 1024. Stąd nazwy wariantów.

### 1.4 Ziarno zamiast macierzy

Macierz $A$ nie musi być tajna, ale musi być losowa. Zamiast ją przesyłać (to byłoby kilka kilobajtów), w kluczu publicznym jest tylko 32-bajtowe ziarno $\rho$. Obie strony wyliczają z niego identyczną macierz funkcją skrótu (rozdział 8).

**Źródła:** [Broeckelmann](https://medium.com/@robert.broeckelmann/post-quantum-cryptography-ml-kem-the-math-7382776f6c44), części „What Is a Lattice?”, „The Hard Problem”, „Why Noise Matters”, „Why Quantum Computers Don't Break It”; [Naskręcki 41.2](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) (Module-LWE); [Lyubashevsky](https://eprint.iacr.org/2024/1287) i [Menezes](https://eprint.iacr.org/2026/1098) (definicje formalne).

---

## 2. Parametry

### 2.1 Co to jest i po co

Wszystkie liczby, od których zależy działanie schematu, są zebrane w `params.py`. Żaden inny moduł nie wpisuje ich na sztywno (CONTRACT §1), bo literówka w jednym miejscu dałaby błędny klucz bez żadnego komunikatu o błędzie.

### 2.2 Stałe wspólne dla wszystkich wariantów

| Stała | Wartość | Znaczenie |
|---|---|---|
| `Q` | 3329 | moduł: wszystkie współczynniki są liczbami od 0 do 3328 |
| `N` | 256 | liczba współczynników jednego wielomianu |
| `ZETA` | 17 | pierwiastek z jedynki potrzebny do NTT (rozdział 5) |
| `N_INV` | 3303 | odwrotność liczby 128 modulo 3329, używana na końcu odwrotnej NTT |

Dlaczego akurat 3329? Ta liczba jest pierwsza, a $3329 - 1 = 3328 = 13 \cdot 256$. Z tego wynika, że modulo 3329 istnieje liczba, której 256. potęga daje 1, a żadna mniejsza nie daje. Taka liczba nazywa się **pierwiastkiem pierwotnym 256. stopnia z jedynki** i jest nią 17. Sprawdzenie w Pythonie: `pow(17, 256, 3329) == 1` oraz `pow(17, 128, 3329) == 3328`, czyli $-1$. Bez takiego pierwiastka nie dałoby się zbudować NTT.

### 2.3 Trzy zestawy parametrów

| Wariant | $k$ | $\eta_1$ | $\eta_2$ | $d_u$ | $d_v$ | Poziom NIST |
|---|---|---|---|---|---|---|
| ML-KEM-512 | 2 | 3 | 2 | 10 | 4 | 1 (jak AES-128) |
| ML-KEM-768 | 3 | 2 | 2 | 10 | 4 | 3 (jak AES-192) |
| ML-KEM-1024 | 4 | 2 | 2 | 11 | 5 | 5 (jak AES-256) |

Co oznacza każdy parametr:

- **$k$**: ile wielomianów ma wektor tajny $s$ i ile wierszy i kolumn ma macierz $A$. Większe $k$ oznacza wyższy poziom bezpieczeństwa, ale też większe klucze.
- **$\eta_1$**: jak duży jest szum w wektorach $s$ i $e$ (przy generowaniu kluczy) oraz $y$ (przy szyfrowaniu). Współczynniki szumu są z przedziału $[-\eta_1, \eta_1]$.
- **$\eta_2$**: to samo dla szumów $e_1$ i $e_2$ dodawanych przy szyfrowaniu.
- **$d_u$**, **$d_v$**: ile bitów zostaje z każdego współczynnika szyfrogramu po kompresji (rozdział 10). Pełny współczynnik ma 12 bitów.

Wariant 512 ma większy szum $\eta_1 = 3$, bo przy mniejszym $k$ zabezpieczeniem jest głównie szum. Większy szum zwiększa bezpieczeństwo, ale też szansę błędnego odszyfrowania, dlatego w pozostałych wariantach jest on mniejszy.

### 2.4 W kodzie

`MLKEMParams` to `dataclass(frozen=True)`, czyli obiekt, którego pól nie można zmienić po utworzeniu. Gotowe zestawy to stałe `ML_KEM_512`, `ML_KEM_768` i `ML_KEM_1024`. Rozmiary kluczy (`ek_size`, `dk_size`, `ct_size`) są wyliczane z $k$, $d_u$ i $d_v$, a nie wpisywane ręcznie (rozdział 14).

**Źródła:** [FIPS 203 §8, tabela 2](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf) (zestawy parametrów); [FIPS 203 §4.3](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf) (definicja $\zeta = 17$); [Naskręcki 41.3](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) (sprawdzenie pierwiastka w kodzie), [41.10](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) (zestawy parametrów); szczegóły kodu: [Params-i-Primitives.md §1](Params-i-Primitives.md).

---

## 3. Pierścień wielomianów $R_q$

### 3.1 Co to jest i po co

Podstawowym obiektem w ML-KEM jest wielomian stopnia co najwyżej 255:

$$f = f_0 + f_1 X + f_2 X^2 + \dots + f_{255} X^{255}$$

Każdy współczynnik $f_i$ jest liczbą modulo 3329. Zbiór takich wielomianów razem z dodawaniem i mnożeniem oznacza się

$$R_q = \mathbb{Z}_q[X] / (X^{256} + 1).$$

Zapis „$/(X^{256}+1)$” mówi, że przy mnożeniu obowiązuje reguła $X^{256} = -1$. Dzięki niej wynik mnożenia znowu ma 256 współczynników.

### 3.2 Jak to działa

**Dodawanie i odejmowanie** robi się współczynnik po współczynniku, a wynik bierze modulo 3329. Przykład: jeśli $f_0 = 3000$ i $g_0 = 500$, to współczynnik sumy wynosi $3500 \bmod 3329 = 171$.

**Mnożenie** robi się jak w szkole (każdy wyraz przez każdy), a potem redukuje potęgi od $X^{256}$ w górę. Z reguły $X^{256} = -1$ wynika

$$X^{256 + j} = X^{256} \cdot X^j = -X^j,$$

czyli wyraz, który wychodzi poza $X^{255}$, wraca na pozycję o 256 niższą **ze zmienionym znakiem**. Takie mnożenie nazywa się **negacyklicznym**.

Przykład na małym pierścieniu z $n = 4$, gdzie $X^4 = -1$:

$$X^3 \cdot X^2 = X^5 = X \cdot X^4 = -X.$$

Przykład w prawdziwym pierścieniu: $X^{100} \cdot X^{200} = X^{300} = -X^{44}$. W kodzie współczynnik przy $X^{44}$ wynosi wtedy $-1 \bmod 3329 = 3328$.

### 3.3 W kodzie

Wielomian to tablica NumPy o kształcie `(256,)` i typie `int64`, gdzie `f[i]` to współczynnik przy $X^i$. Wartości są zawsze w przedziale `[0, 3329)`.

| Funkcja | Działanie |
|---|---|
| `poly_add(a, b)` | `(a + b) % Q` |
| `poly_sub(a, b)` | `(a - b) % Q`; w Pythonie i NumPy wynik `%` jest zawsze nieujemny |
| `schoolbook_mul(a, b)` | mnożenie szkolne z regułą $X^{256} = -1$ |

`schoolbook_mul` liczy pełny iloczyn o 512 współczynnikach, a potem odejmuje górną połowę od dolnej: `(c[:256] - c[256:]) % Q`. To odejmowanie to właśnie reguła $X^{256+j} = -X^j$.

### 3.4 Na co uważać

- **Typ `int64` jest konieczny.** Jeden współczynnik iloczynu to suma do 256 składników, każdy do $3328^2$. Razem to około $2{,}8 \cdot 10^9$, więcej niż mieści `int32`. NumPy nie zgłasza przepełnienia, tylko po cichu daje zły wynik.
- **Mnożenie szkolne jest wolne:** $256^2 = 65\,536$ mnożeń na jeden iloczyn. Dlatego w docelowej wersji używa się NTT (rozdział 5), a `schoolbook_mul` służy jako wzorzec w testach.

**Źródła:** [FIPS 203 §2.4.4–2.4.5](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf); [Naskręcki 41.2](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html); [Broeckelmann, „Polynomial Arithmetic”](https://medium.com/@robert.broeckelmann/post-quantum-cryptography-ml-kem-the-math-7382776f6c44); szczegóły kodu i testów: [Polynomial.md](Polynomial.md).

---

## 4. Wektory i macierze wielomianów

### 4.1 Co to jest i po co

W Module-LWE zamiast liczb są wielomiany. Wektor tajny $s$ to $k$ wielomianów, a macierz $A$ to $k \times k$ wielomianów. Wszystkie wzory wyglądają jak w zwykłej algebrze liniowej, tylko „mnożenie liczb” zastępuje mnożenie wielomianów w $R_q$.

### 4.2 Jak to działa

Iloczyn macierzy przez wektor: $i$-ty element wyniku to suma iloczynów $i$-tego wiersza macierzy z wektorem.

$$(A \cdot s)_i = \sum_{j=0}^{k-1} A_{i,j} \cdot s_j$$

Iloczyn skalarny dwóch wektorów daje jeden wielomian:

$$t^\top \cdot y = \sum_{i=0}^{k-1} t_i \cdot y_i$$

Transpozycja $A^\top$ zamienia wiersze z kolumnami: $(A^\top)_{i,j} = A_{j,i}$.

Przykład dla $k = 2$:

$$A \cdot s = \begin{pmatrix} A_{0,0} \cdot s_0 + A_{0,1} \cdot s_1 \\ A_{1,0} \cdot s_0 + A_{1,1} \cdot s_1 \end{pmatrix}$$

Każde „$\cdot$” w tym wzorze to mnożenie dwóch wielomianów, a każde „$+$” to `poly_add`.

### 4.3 W kodzie

| Obiekt | Kształt tablicy | Przykład |
|---|---|---|
| wielomian | `(256,)` | `e2`, `v`, `mu` |
| wektor | `(k, 256)` | `s`, `e`, `y`, `u` |
| macierz | `(k, k, 256)` | `a_hat` |

Ostatni wymiar to zawsze współczynniki wielomianu, a wcześniejsze wymiary to indeksy wektora lub macierzy. Funkcje w `polynomial.py`:

- `dot(u_hat, v_hat)`: iloczyn skalarny w postaci NTT,
- `matvec(a_hat, v_hat)`: macierz razy wektor w postaci NTT,
- `transpose(a_hat)`: transpozycja macierzy wielomianów.

### 4.4 Na co uważać

- **`*` i `@` z NumPy nie mnożą wielomianów.** `a * b` mnoży współczynnik po współczynniku, co w zwykłej postaci jest błędne. Trzeba użyć `schoolbook_mul` albo `multiply_ntts`.
- **Transpozycja to `np.transpose(a, (1, 0, 2))`, a nie `a.T`.** Samo `.T` odwróciłoby wszystkie trzy wymiary i przestawiło też współczynniki wielomianów.

**Źródła:** [FIPS 203 §2.4.6–2.4.7](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf); [Naskręcki 41.14](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html); [Broeckelmann, „Matrix Multiplication with Polynomials”](https://medium.com/@robert.broeckelmann/post-quantum-cryptography-ml-kem-the-math-7382776f6c44).

---

## 5. NTT: szybkie mnożenie

### 5.1 Co to jest i po co

NTT (*Number-Theoretic Transform*) to przekształcenie wielomianu do innej postaci, w której mnożenie jest dużo prostsze. Zamiast $65\,536$ mnożeń wystarcza kilkaset. To odpowiednik szybkiej transformaty Fouriera, tylko na liczbach modulo $q$ zamiast na liczbach zespolonych.

W ML-KEM NTT nie jest tylko optymalizacją. **Standard wymaga, żeby macierz $\hat{A}$ i wektor $\hat{t}$ w kluczu publicznym były zapisane w postaci NTT.** Bez NTT klucze nie będą zgodne z innymi implementacjami.

### 5.2 Intuicja

Wielomian można opisać na dwa sposoby: przez jego współczynniki albo przez jego wartości w wybranych punktach. W drugim opisie mnożenie jest proste, bo wystarczy pomnożyć wartości w każdym punkcie osobno. NTT przechodzi z pierwszego opisu do drugiego, a odwrotna NTT wraca.

W ML-KEM jest jedna komplikacja. Pełne rozbicie wymagałoby pierwiastka 512. stopnia z jedynki, a modulo 3329 takiego nie ma ($512$ nie dzieli $3328$). Dlatego NTT zatrzymuje się wcześniej i zamiast 256 pojedynczych wartości daje **128 par**.

### 5.3 Jak to działa

Wielomian $X^{256} + 1$ rozkłada się na 128 czynników stopnia 2:

$$X^{256} + 1 = \prod_{i=0}^{127} \left( X^2 - \gamma_i \right), \qquad \gamma_i = 17^{2 \cdot \mathrm{BitRev}_7(i) + 1} \bmod q.$$

$\mathrm{BitRev}_7(i)$ to liczba $i$ zapisana na 7 bitach i odczytana od tyłu. Na przykład $i = 1 = 0000001_2$ daje $1000000_2 = 64$.

NTT wielomianu $f$ to 128 reszt z dzielenia $f$ przez kolejne czynniki $X^2 - \gamma_i$. Każda reszta ma postać $a + bX$, czyli dwie liczby, więc wynik nadal ma 256 liczb.

Algorytm nie dzieli wielomianów wprost. Robi to w **7 warstwach**. W każdej warstwie tablica jest podzielona na bloki, a każda para elementów odległych o `length` przechodzi przez „motylek”:

```text
t          = zeta * f[j + length]      (mod q)
f[j+length] = f[j] - t                 (mod q)
f[j]        = f[j] + t                 (mod q)
```

Krok po kroku:

1. Warstwa 1: `length = 128`. Jeden blok, element `j` łączy się z `j + 128`.
2. Warstwa 2: `length = 64`. Dwa bloki, w każdym element łączy się z odległym o 64.
3. Kolejne warstwy: `length = 32, 16, 8, 4, 2`.
4. Każdy blok używa innego mnożnika `zeta = ZETAS[k]`, gdzie `k` rośnie od 1 do 127.

Po 7 warstwach mamy 128 par, czyli gotową postać NTT.

**Odwrotna NTT** wykonuje te same warstwy w odwrotnej kolejności (`length = 2, 4, ..., 128`), z `k` malejącym od 127, a na końcu mnoży wszystko przez $128^{-1} \bmod q = 3303$. Sprawdzenie: $128 \cdot 3303 = 422\,784 = 127 \cdot 3329 + 1$.

### 5.4 W kodzie

- `ZETAS[i] = 17^BitRev7(i) mod Q`: mnożniki motylków. Pierwsze wartości to `1, 1729, 2580, 3289, 2642, 630, 1897, 848`, zgodne z FIPS 203, dodatek A.
- `GAMMAS[i] = 17^(2·BitRev7(i)+1) mod Q`: stałe $\gamma_i$ do mnożenia (rozdział 6). Pierwsze wartości: `17, 3312, 2761, 568`.
- `ntt(f)` (Alg. 9) i `inv_ntt(f_hat)` (Alg. 10) przyjmują i zwracają tablice `(256,)` i nie zmieniają wejścia.
- `ntt_vec` i `inv_ntt_vec` stosują je do każdego wielomianu wektora `(k, 256)`.

Tablica w postaci NTT ma ten sam kształt `(256,)`, ale inne znaczenie: elementy `2i` i `2i+1` to współczynniki pary numer $i$. Dlatego takie zmienne mają w nazwie `_hat`.

### 5.5 Na co uważać

- `k` w NTT zaczyna od 1, a nie od 0 (`ZETAS[0] = 1` nie jest używane).
- Na końcu odwrotnej NTT mnoży się przez odwrotność 128, a nie 256, bo warstw jest 7, a nie 8.
- Testy, które sprawdzają poprawność: `inv_ntt(ntt(f)) == f` oraz `inv_ntt(multiply_ntts(ntt(f), ntt(g))) == schoolbook_mul(f, g)`.
- **W naszym kodzie NTT nie jest szybsze od mnożenia szkolnego.** `ntt` i `inv_ntt` to zwykłe pętle Pythona po pojedynczych liczbach, a `schoolbook_mul` ma tylko jedną pętlę, bo resztę robi NumPy w C. Pomiar: jedno mnożenie przez NTT trwa ok. 2 ms, a szkolne ok. 0,8 ms. Mniejsza liczba działań daje przewagę dopiero w językach kompilowanych. U nas NTT jest potrzebne przede wszystkim dlatego, że wymaga go format kluczy, a kod ma być jak najprostszy, więc nie optymalizujemy go.

**Źródła:** [FIPS 203 §4.3, Alg. 9 i 10, dodatek A](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf); [Naskręcki 41.3–41.4](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) (kod w NumPy, diagram motylków, porównanie czasu ze zwykłym mnożeniem), [41.13](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) (wizualizacja mnożników); [RareSkills: Number Theoretic Transform](https://rareskills.io/post/number-theoretic-transform) (intuicja). RareSkills opisuje pełne NTT na małym przykładzie, bez wariantu negacyklicznego i bez par, więc do szczegółów ML-KEM trzeba używać FIPS 203.

---

## 6. Mnożenie w postaci NTT

### 6.1 Co to jest i po co

Po NTT oba wielomiany to 128 par. Żeby je pomnożyć, mnoży się parę numer $i$ pierwszego wielomianu przez parę numer $i$ drugiego, każdą osobno.

### 6.2 Jak to działa

Para to wielomian stopnia 1: $a_0 + a_1 X$. Mnożymy dwie pary i redukujemy modulo $X^2 - \gamma$, czyli stosujemy regułę $X^2 = \gamma$:

$$(a_0 + a_1 X)(b_0 + b_1 X) = a_0 b_0 + (a_0 b_1 + a_1 b_0) X + a_1 b_1 X^2.$$

Po podstawieniu $X^2 = \gamma$:

$$c_0 = a_0 b_0 + a_1 b_1 \gamma, \qquad c_1 = a_0 b_1 + a_1 b_0.$$

Przykład dla $\gamma = 17$: $(1 + 2X)(3 + 4X)$ daje $c_0 = 1 \cdot 3 + 2 \cdot 4 \cdot 17 = 139$ i $c_1 = 1 \cdot 4 + 2 \cdot 3 = 10$.

### 6.3 W kodzie

- `base_case_multiply(a0, a1, b0, b1, gamma)` (Alg. 12) liczy jedną parę według wzoru powyżej.
- `multiply_ntts(f_hat, g_hat)` (Alg. 11) wywołuje ją dla każdej z 128 par z odpowiednim `GAMMAS[i]`.
- `dot` i `matvec` sumują wyniki `multiply_ntts`, więc pracują w postaci NTT.

### 6.4 Na co uważać

Każda para ma inne $\gamma$. Użycie jednego $\gamma$ dla wszystkich par da wynik, który wygląda sensownie, ale jest błędny.

**Źródła:** [FIPS 203 §4.3.1, Alg. 11 i 12](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf); [Naskręcki 41.3, „Pointwise Multiplication”](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html).

---

## 7. Funkcje skrótu

### 7.1 Co to jest i po co

ML-KEM potrzebuje dużo bajtów, które wyglądają na losowe, ale dają się **odtworzyć** z krótkiego ziarna. Odbiorca musi np. wyliczyć dokładnie tę samą macierz $A$ co nadawca. Do tego służą funkcje skrótu z rodziny SHA-3. To samo wejście daje zawsze to samo wyjście, a wyjście nie zdradza nic o wejściu.

### 7.2 Pięć funkcji

| Funkcja | Realizacja | Wyjście | Do czego służy |
|---|---|---|---|
| $H(x)$ | SHA3-256 | 32 B | skrót klucza publicznego $H(ek)$ |
| $G(x)$ | SHA3-512, dzielony na 32 + 32 B | 2 × 32 B | KeyGen: $(\rho, \sigma) = G(d \| k)$; Encaps: $(K, r) = G(m \| H(ek))$ |
| $J(x)$ | SHAKE256 | 32 B | klucz zastępczy przy odrzuceniu: $\bar{K} = J(z \| c)$ |
| $\mathrm{PRF}_\eta(s, b)$ | SHAKE256$(s \| b)$ | $64\eta$ B | bajty do losowania szumu |
| $\mathrm{XOF}(x)$ | SHAKE128 | dowolna długość | bajty do losowania macierzy $\hat{A}$ |

SHA3 daje wynik o stałej długości, a SHAKE o dowolnej. Wszystkie są zbudowane na tej samej funkcji Keccak.

### 7.3 W kodzie

Wszystko jest w `crypto_primitives.py` i korzysta z modułu `hashlib` z biblioteki standardowej, np. `hashlib.sha3_256(data).digest()` albo `hashlib.shake_128(seed).digest(length)`.

### 7.4 Na co uważać

- **Bajt $k$ w $G(d \| k)$** jest wymagany przez ostateczną wersję FIPS 203. Bez niego kod działa sam ze sobą, ale daje inne klucze niż standard.
- **`bytes([b])` to nie to samo co `bytes(b)`.** Pierwsze daje jeden bajt o wartości `b`, drugie `b` bajtów zerowych. Pomyłka nie zgłasza błędu.
- **Licznik $b$ w PRF musi być inny dla każdego wielomianu szumu.** Powtórzenie licznika daje dwa identyczne szumy, co osłabia bezpieczeństwo.
- **`hashlib` nie ma trybu strumieniowego.** Standard opisuje XOF jako strumień, z którego „wyciska się” kolejne bajty. Zamiast tego prosimy o konkretną liczbę bajtów. Działa to, bo dłuższy wynik SHAKE zaczyna się od krótszego: `shake_128(x).digest(8)` zaczyna się od `shake_128(x).digest(4)`.

**Źródła:** [FIPS 203 §4.1](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf) (definicje funkcji), dodatek C (zmiana $G(d)$ na $G(d \| k)$); [dokumentacja `hashlib`](https://docs.python.org/3/library/hashlib.html); [Breaking down SHA-3](https://infosecwriteups.com/breaking-down-sha-3-algorithm-70fe25e125b6) (budowa SHA-3 w środku; nie omawia SHAKE); szczegóły kodu: [Params-i-Primitives.md §2](Params-i-Primitives.md).

---

## 8. Losowanie macierzy: SampleNTT

### 8.1 Co to jest i po co

Macierz $\hat{A}$ ma wyglądać na całkowicie losową, ale obie strony muszą ją wyliczyć identycznie z ziarna $\rho$. `sample_ntt` tworzy jeden jej element: wielomian o 256 współczynnikach rozłożonych **równomiernie** na liczbach od 0 do 3328. Wynik jest od razu traktowany jako wielomian w postaci NTT.

### 8.2 Jak to działa

1. Wejściem XOF jest $\rho \| j \| i$, czyli ziarno i dwa bajty z indeksami. Element $\hat{A}_{i,j}$ powstaje więc z innych bajtów niż $\hat{A}_{j,i}$.
2. Strumień bajtów czyta się po 3 bajty $b_0, b_1, b_2$ (24 bity) i robi się z nich dwie liczby 12-bitowe:
   $$d_1 = b_0 + 256 \cdot (b_1 \bmod 16), \qquad d_2 = \lfloor b_1 / 16 \rfloor + 16 \cdot b_2.$$
3. Liczba 12-bitowa może mieć wartość od 0 do 4095. Jeśli jest mniejsza niż 3329, trafia do wyniku. Jeśli nie, jest odrzucana.
4. Kroki 2 i 3 powtarza się, aż zbierze się 256 współczynników.

Przykład: bajty `23 C1 AB` dają $d_1 = 0x23 + 256 \cdot 1 = 291$ i $d_2 = 12 + 16 \cdot 171 = 2748$. Obie są mniejsze niż 3329, więc obie trafiają do wyniku.

**Dlaczego odrzucać, a nie liczyć modulo 3329?** Gdyby brać $d \bmod 3329$, liczby od 0 do 766 pojawiałyby się dwa razy częściej niż pozostałe (raz bezpośrednio, raz jako $3329$–$4095$). Odrzucanie daje rozkład idealnie równomierny. Ta metoda nazywa się **próbkowaniem z odrzucaniem** (*rejection sampling*).

### 8.3 W kodzie

`sample_ntt(rho, i, j)` w `sampling.py`. Sklejenie w kolejności `rho + bytes([j, i])` jest w środku funkcji, więc wywołujący pisze po prostu `a_hat[i, j] = sample_ntt(rho, i, j)` (CONTRACT §5.1). Funkcja prosi najpierw o 840 bajtów (280 trójek, czyli 560 kandydatów), co prawie zawsze wystarcza, bo przyjmowanych jest około 81% kandydatów. Jeśli nie wystarczy, prosi o dwa razy więcej.

### 8.4 Na co uważać

Zamiana kolejności $i$ i $j$ jest najczęstszym błędem. Szyfrowanie i odszyfrowanie nadal działają wtedy same ze sobą, bo obie strony używają tej samej (przetransponowanej) macierzy, ale klucze są niezgodne z kyber-py. Wykrywa to test porównujący całą macierz z kyber-py.

**Źródła:** [FIPS 203 §4.2.2, Alg. 7](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf), dodatek B (ile bajtów XOF może być potrzebne); [Naskręcki 41.5](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html).

---

## 9. Losowanie szumu: SamplePolyCBD

### 9.1 Co to jest i po co

Szum ($s$, $e$, $y$, $e_1$, $e_2$) musi mieć małe współczynniki, skupione wokół zera. Używa się do tego **centralnego rozkładu dwumianowego** CBD$_\eta$ (*centered binomial distribution*), bo łatwo go wylosować z bitów bez żadnych liczb zmiennoprzecinkowych.

### 9.2 Jak to działa

Każdy współczynnik zużywa $2\eta$ losowych bitów:

1. Bierze się $\eta$ bitów i liczy ich sumę $x$ (liczba jedynek).
2. Bierze się kolejne $\eta$ bitów i liczy ich sumę $y$.
3. Współczynnik to $x - y$, czyli liczba z przedziału $[-\eta, \eta]$. Wartość ujemną zapisuje się modulo $q$, np. $-1$ jako $3328$.

Na 256 współczynników potrzeba $256 \cdot 2\eta$ bitów, czyli dokładnie $64\eta$ bajtów, tyle ile daje $\mathrm{PRF}_\eta$.

Przykład dla $\eta = 2$: bajt `10110111` czytany od najmłodszego bitu daje bity `1,1,1,0,1,1,0,1`. Pierwszy współczynnik: $x = 1 + 1 = 2$, $y = 1 + 0 = 1$, wynik $1$. Drugi: $x = 1 + 1 = 2$, $y = 0 + 1 = 1$, wynik $1$.

Rozkład dla $\eta = 2$:

| Wartość | $-2$ | $-1$ | $0$ | $1$ | $2$ |
|---|---|---|---|---|---|
| Prawdopodobieństwo | 1/16 | 4/16 | 6/16 | 4/16 | 1/16 |

### 9.3 W kodzie

`sample_poly_cbd(eta, buf)` w `sampling.py`. Bity są wyciągane funkcją `bytes_to_bits` (najmłodszy bit bajtu jest pierwszy), a potem przekształcane w tablicę `(256, 2·eta)` i sumowane wierszami, bez pętli po bitach.

Każdy wielomian szumu dostaje bajty z innym licznikiem PRF:

- **KeyGen:** $s_0, \dots, s_{k-1}$ z liczników $0, \dots, k-1$, potem $e_0, \dots, e_{k-1}$ z liczników $k, \dots, 2k-1$. Wszystko z ziarna $\sigma$ i z $\eta_1$.
- **Encrypt:** najpierw $y$ (z $\eta_1$), potem $e_1$ (z $\eta_2$), na końcu $e_2$ (z $\eta_2$). Licznik rośnie przez cały czas od 0 do $2k$. Wszystko z ziarna $r$.

### 9.4 Na co uważać

Licznik jest wspólny dla kolejnych wektorów, a nie zerowany przed każdym z nich. Zerowanie dałoby $e = s$, czyli klucz publiczny bez żadnej ochrony.

**Źródła:** [FIPS 203 §4.2.2, Alg. 8](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf), Alg. 13 i 14 (kolejność liczników); [Naskręcki 41.5](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html).

---

## 10. Kompresja

### 10.1 Co to jest i po co

Kompresja służy do dwóch rzeczy:

1. **Zmniejsza szyfrogram.** Zamiast 12 bitów na współczynnik zostaje $d_u$ albo $d_v$ bitów. Utracona dokładność działa jak dodatkowy mały szum, który schemat i tak toleruje.
2. **Zamienia bity wiadomości na współczynniki i z powrotem** (wersja z $d = 1$).

### 10.2 Jak to działa

Liczby od 0 do 3328 można sobie wyobrazić na okręgu, bo po 3328 znowu jest 0. Kompresja do $d$ bitów dzieli ten okrąg na $2^d$ równych odcinków i zapamiętuje tylko numer najbliższego punktu podziału:

$$\mathrm{Compress}_d(x) = \left\lfloor \frac{2^d}{q} \cdot x \right\rceil \bmod 2^d$$

Dekompresja zamienia numer z powrotem na liczbę z zakresu $[0, q)$:

$$\mathrm{Decompress}_d(y) = \left\lfloor \frac{q}{2^d} \cdot y \right\rceil$$

Nawias $\lfloor \cdot \rceil$ oznacza zaokrąglenie do najbliższej liczby całkowitej. Kompresja jest stratna: $\mathrm{Decompress}(\mathrm{Compress}(x))$ jest tylko blisko $x$.

Przykłady (policzone na naszym kodzie):

| $x$ | $d$ | Compress | Decompress | Błąd |
|---|---|---|---|---|
| 1000 | 10 | 308 | 1001 | 1 |
| 1000 | 4 | 5 | 1040 | 40 |

**Przypadek $d = 1$ (wiadomość).** $\mathrm{Decompress}_1$ zamienia bit 0 na 0, a bit 1 na 1665, czyli mniej więcej $q/2$. Te dwie wartości leżą na okręgu naprzeciw siebie. $\mathrm{Compress}_1$ działa odwrotnie: daje 1 dla $x$ od 833 do 2496 i 0 dla pozostałych (bliżej 0 lub bliżej 3329). Dzięki temu bit przetrwa każdy szum mniejszy niż około $q/4 \approx 832$.

### 10.3 W kodzie

Plik `compress.py`. Wzory liczy się wyłącznie na liczbach całkowitych:

```python
compress: ((x << d) + Q // 2) // Q & ((1 << d) - 1)
decompress: (Q * y + (1 << (d - 1))) // (1 << d)
```

`x << d` to mnożenie przez $2^d$, a `// (1 << d)` to dzielenie przez $2^d$ z zaokrągleniem w dół. Dodanie połowy dzielnika przed dzieleniem daje zaokrąglenie do najbliższej. `& ((1 << d) - 1)` zostawia najmłodsze $d$ bitów, czyli liczy resztę z dzielenia przez $2^d$.

### 10.4 Na co uważać

- **Nie używać liczb zmiennoprzecinkowych.** `round(2**d / Q * x)` czasem zaokrągla inaczej niż standard, co przy niektórych wartościach daje inny bajt w szyfrogramie.
- **Reszta z dzielenia przez $2^d$ na końcu jest potrzebna.** Np. dla $x = 3328$ i $d = 10$ zaokrąglenie daje 1024, czyli poza zakresem, a po redukcji 0. Na okręgu 3328 jest tuż obok 0.

**Źródła:** [FIPS 203 §4.2.1, wzory 4.7 i 4.8](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf); [Naskręcki 41.7](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) (kompresja), [41.9](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) i [41.15](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) (dlaczego szum nie psuje odszyfrowania); dopuszczalne błędy dla każdego $d$: [CONTRACT.md](CONTRACT.md).

---

## 11. Zamiana na bajty: ByteEncode i ByteDecode

### 11.1 Co to jest i po co

Klucze i szyfrogram przesyła się jako ciągi bajtów. `byte_encode` zamienia wielomian, czyli 256 liczb po $d$ bitów, na $32d$ bajtów. `byte_decode` robi to odwrotnie.

### 11.2 Jak to działa

1. Każdą z 256 liczb zapisuje się na $d$ bitach, **zaczynając od najmłodszego bitu** (*little-endian*).
2. Wszystkie bity skleja się w jeden ciąg $256 \cdot d$ bitów.
3. Ciąg dzieli się na bajty po 8 bitów. Pierwszy bit ciągu staje się najmłodszym bitem pierwszego bajtu.

Przykład dla $d = 1$: współczynniki `1, 0, 1, 1, 0, 0, 0, 0` dają bajt `00001101`, czyli `0x0d`.

Przykład dla $d = 12$: dwie liczby $a = \mathtt{0x123}$ i $b = \mathtt{0xABC}$ zajmują razem 24 bity, czyli 3 bajty: `23 C1 AB`. Pierwszy bajt to młodsze 8 bitów $a$, drugi łączy 4 starsze bity $a$ i 4 młodsze bity $b$, trzeci to 8 starszych bitów $b$. Ten sam układ odczytuje `sample_ntt` (rozdział 8).

### 11.3 W kodzie

W `compress.py`: `bits_to_bytes` (Alg. 3), `bytes_to_bits` (Alg. 4), `byte_encode(d, f)` (Alg. 5), `byte_decode(d, b)` (Alg. 6). Wejście `byte_encode` to tablica `(256,)`, a wyjście to `bytes` o długości `32 * d`.

### 11.4 Na co uważać

- Dla $d = 12$ liczba 12-bitowa może mieć wartość do 4095, a współczynnik tylko do 3328. Dlatego `byte_decode(12, …)` redukuje wynik modulo $q$. To samo zjawisko wykorzystuje kontrola klucza publicznego: jeśli `byte_encode(12, byte_decode(12, ek))` różni się od `ek`, klucz zawierał niepoprawne wartości (rozdział 13.5).
- Jeden przestawiony bit daje zupełnie inny klucz, dlatego kodowanie testuje się w obie strony i porównuje z kyber-py.

**Źródła:** [FIPS 203 §4.2.1, Alg. 3–6](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf). Naskręcki w uproszczonej implementacji pomija to kodowanie, więc jedynym źródłem jest standard.

---

## 12. K-PKE: szyfrowanie

K-PKE to szyfrowanie z kluczem publicznym, które szyfruje dokładnie 32 bajty (256 bitów). ML-KEM używa go do przesłania losowej wiadomości $m$, z której potem powstaje klucz $K$.

### 12.1 Generowanie kluczy: K-PKE.KeyGen (Alg. 13)

**Wejście:** 32-bajtowe ziarno $d$. **Wyjście:** klucz publiczny $ek_{pke}$ i prywatny $dk_{pke}$.

1. Z ziarna powstają dwa nowe: $(\rho, \sigma) = G(d \| k)$. Ziarno $\rho$ jest publiczne i posłuży do macierzy, $\sigma$ jest tajne i posłuży do szumu.
2. Losuje się macierz $\hat{A}$: dla każdej pary $(i, j)$ element $\hat{A}_{i,j} = \mathrm{SampleNTT}(\rho, i, j)$.
3. Losuje się tajny wektor $s$ i wektor błędu $e$ z rozkładu CBD$_{\eta_1}$, z bajtów $\mathrm{PRF}(\sigma, 0), \mathrm{PRF}(\sigma, 1), \dots$
4. Oba wektory przechodzą przez NTT: $\hat{s} = \mathrm{NTT}(s)$, $\hat{e} = \mathrm{NTT}(e)$.
5. Liczy się część publiczną klucza: $\hat{t} = \hat{A} \circ \hat{s} + \hat{e}$. To jest instancja problemu LWE z rozdziału 1.
6. Klucz publiczny to $ek_{pke} = \mathrm{ByteEncode}_{12}(\hat{t}) \| \rho$, a klucz prywatny to $dk_{pke} = \mathrm{ByteEncode}_{12}(\hat{s})$.

### 12.2 Szyfrowanie: K-PKE.Encrypt (Alg. 14)

**Wejście:** klucz publiczny $ek_{pke}$, wiadomość $m$ (32 bajty) i 32 bajty losowości $r$. **Wyjście:** szyfrogram $c$.

1. Z klucza odczytuje się $\hat{t}$ (`byte_decode(12, …)`) i $\rho$ (ostatnie 32 bajty), a z $\rho$ odtwarza tę samą macierz $\hat{A}$ co w KeyGen.
2. Z bajtów $\mathrm{PRF}(r, \cdot)$ losuje się trzy szumy: wektor $y$ z CBD$_{\eta_1}$, wektor $e_1$ i wielomian $e_2$ z CBD$_{\eta_2}$.
3. Liczy się pierwszą część szyfrogramu: $u = \mathrm{NTT}^{-1}(\hat{A}^\top \circ \hat{y}) + e_1$, gdzie $\hat{y} = \mathrm{NTT}(y)$.
4. Wiadomość zamienia się na wielomian: $\mu = \mathrm{Decompress}_1(\mathrm{ByteDecode}_1(m))$. Każdy bit 0 staje się współczynnikiem 0, a bit 1 współczynnikiem 1665.
5. Liczy się drugą część: $v = \mathrm{NTT}^{-1}(\hat{t}^\top \circ \hat{y}) + e_2 + \mu$.
6. Obie części kompresuje się i zamienia na bajty: $c = \mathrm{ByteEncode}_{d_u}(\mathrm{Compress}_{d_u}(u)) \| \mathrm{ByteEncode}_{d_v}(\mathrm{Compress}_{d_v}(v))$.

Szum $e_1$ i $e_2$ dodaje się już w zwykłej postaci, bez NTT.

### 12.3 Odszyfrowanie: K-PKE.Decrypt (Alg. 15)

**Wejście:** klucz prywatny $dk_{pke}$ i szyfrogram $c$. **Wyjście:** wiadomość $m'$.

1. Szyfrogram dzieli się na dwie części i odtwarza z nich $u$ i $v$ przez `byte_decode` i `decompress`.
2. Z klucza prywatnego odczytuje się $\hat{s}$.
3. Liczy się $w = v - \mathrm{NTT}^{-1}(\hat{s}^\top \circ \mathrm{NTT}(u))$.
4. Wiadomość to $m' = \mathrm{ByteEncode}_1(\mathrm{Compress}_1(w))$: każdy współczynnik bliski 1665 daje bit 1, a bliski 0 daje bit 0.

### 12.4 Dlaczego odszyfrowanie działa

Rozpiszmy $w$ bez NTT, bo NTT nie zmienia wyniku, tylko sposób liczenia. Klucz publiczny to $t = A s + e$, więc $t^\top = s^\top A^\top + e^\top$. Podstawiamy:

$$
\begin{aligned}
w &= v - s^\top u \\
  &= (t^\top y + e_2 + \mu) - s^\top (A^\top y + e_1) \\
  &= s^\top A^\top y + e^\top y + e_2 + \mu - s^\top A^\top y - s^\top e_1 \\
  &= \mu + \underbrace{e^\top y + e_2 - s^\top e_1}_{\text{mały szum}}
\end{aligned}
$$

Duże składniki $s^\top A^\top y$ się skracają, bo odbiorca zna $s$. Zostaje wiadomość $\mu$ plus suma iloczynów małych wektorów, do której dochodzi jeszcze błąd kompresji. Ten łączny szum jest dużo mniejszy niż 832 (w naszych testach największa wartość wynosiła około 200–260), więc $\mathrm{Compress}_1$ poprawnie rozpoznaje każdy bit.

Ktoś, kto nie zna $s$, nie może tego skrócenia wykonać. Dla niego $u$ i $v$ wyglądają jak losowe liczby, bo to znowu instancje LWE.

### 12.5 W kodzie

`k_pke_keygen(params, d)`, `k_pke_encrypt(params, ek_pke, m, r)` i `k_pke_decrypt(params, dk_pke, c)` w `pke.py`. Wszystkie przyjmują i zwracają `bytes`.

### 12.6 Na co uważać

- $u$ używa macierzy **transponowanej** $\hat{A}^\top$, a $\hat{t}$ jej nie transponowanej. Pomylenie tego psuje odszyfrowanie.
- Samo K-PKE nie jest bezpieczne wobec aktywnego atakującego. Ktoś, kto może wysyłać zmienione szyfrogramy i obserwować, czy odszyfrowanie się udało, może stopniowo odtworzyć klucz prywatny. Przed tym chroni dopiero ML-KEM.

**Źródła:** [FIPS 203 §5, Alg. 13–15](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf); [Naskręcki 41.9](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) (analiza szumu przy odszyfrowaniu), [41.15](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) („Why Decryption Works”); [Lyubashevsky](https://eprint.iacr.org/2024/1287) (wzory szyfrowania opartego na LWE); [CrypTool](https://www.cryptool.org/en/cto/) (interaktywne przejście przez kroki).

---

## 13. ML-KEM: wymiana klucza i transformacja Fujisakiego-Okamoto

### 13.1 Co to jest i po co

ML-KEM korzysta z K-PKE, ale dodaje do niego **transformację Fujisakiego-Okamoto** (FO). Celem jest ochrona przed atakiem, w którym atakujący wysyła spreparowane szyfrogramy do odbiorcy i obserwuje jego reakcję. Pomysł FO polega na tym, że odbiorca **sprawdza, czy szyfrogram powstał uczciwie**, i jeśli nie, zwraca bezużyteczny klucz zamiast komunikatu o błędzie.

Kluczowa zmiana w stosunku do K-PKE: losowość szyfrowania $r$ nie jest wybierana dowolnie, tylko wyliczana z wiadomości. Dzięki temu odbiorca po odszyfrowaniu wiadomości może powtórzyć całe szyfrowanie i porównać wynik.

### 13.2 Generowanie kluczy: ML-KEM.KeyGen (Alg. 16 i 19)

1. Losuje się dwa 32-bajtowe ziarna: $d$ (do kluczy) i $z$ (do odrzucania, punkt 13.4).
2. $(ek_{pke}, dk_{pke}) = \mathrm{K\text{-}PKE.KeyGen}(d)$.
3. Klucz publiczny to $ek = ek_{pke}$.
4. Klucz prywatny to sklejenie czterech części: $dk = dk_{pke} \| ek \| H(ek) \| z$.

Klucz prywatny zawiera kopię klucza publicznego i jego skrót, bo przy odbiorze trzeba będzie powtórzyć szyfrowanie.

### 13.3 Enkapsulacja: ML-KEM.Encaps (Alg. 17 i 20)

1. Losuje się 32-bajtową wiadomość $m$.
2. Z wiadomości i skrótu klucza powstają dwie wartości: $(K, r) = G(m \| H(ek))$. $K$ to przyszły wspólny klucz, a $r$ to losowość do szyfrowania.
3. Szyfruje się wiadomość: $c = \mathrm{K\text{-}PKE.Encrypt}(ek, m, r)$.
4. Wynik: klucz $K$ zostaje u nadawcy, szyfrogram $c$ idzie do odbiorcy.

Ponieważ $r$ zależy tylko od $m$ i $ek$, ta sama wiadomość zawsze daje ten sam szyfrogram.

### 13.4 Dekapsulacja: ML-KEM.Decaps (Alg. 18 i 21)

1. Z $dk$ wycina się cztery części: $dk_{pke}$, $ek$, $h = H(ek)$ i $z$.
2. Odszyfrowuje się wiadomość: $m' = \mathrm{K\text{-}PKE.Decrypt}(dk_{pke}, c)$.
3. Liczy się kandydata na klucz i losowość: $(K', r') = G(m' \| h)$.
4. Liczy się klucz zastępczy: $\bar{K} = J(z \| c)$.
5. **Ponownie szyfruje się wiadomość:** $c' = \mathrm{K\text{-}PKE.Encrypt}(ek, m', r')$.
6. Porównuje się szyfrogramy:
   - jeśli $c = c'$, szyfrogram jest uczciwy i wynikiem jest $K'$,
   - jeśli $c \ne c'$, wynikiem jest $\bar{K}$.

Jeśli szyfrogram jest uczciwy, to $m' = m$, więc $K' = K$ i obie strony mają ten sam klucz.

Jeśli atakujący zmienił choć jeden bit $c$, to ponowne szyfrowanie da inny $c'$ i odbiorca zwróci $\bar{K}$. Ten klucz wygląda na losowy, ale jest zawsze taki sam dla tego samego $c$, bo zależy od tajnego $z$. Atakujący nie dostaje komunikatu o błędzie i nie potrafi odróżnić $\bar{K}$ od prawdziwego klucza. Taka reakcja nazywa się **niejawnym odrzuceniem** (*implicit rejection*). Połączenie rozmowy po prostu się nie uda, ale atakujący nie dowie się z tego niczego o kluczu prywatnym.

### 13.5 Kontrola danych wejściowych (FIPS 203 §7.2 i §7.3)

Wersje docelowe (Alg. 20 i 21) przed właściwym działaniem sprawdzają dane:

- **Encaps:** czy $ek$ ma właściwą długość i czy wszystkie zakodowane współczynniki są mniejsze niż $q$. Sprawdzenie: `byte_encode(12, byte_decode(12, część_ek)) == część_ek`.
- **Decaps:** czy $c$ i $dk$ mają właściwe długości oraz czy skrót zapisany w $dk$ zgadza się z $H(ek)$ dla zapisanego tam $ek$.

Błędne dane kończą się wyjątkiem `ValueError` (CONTRACT §3). W kodzie robią to `ml_kem_encaps` i `ml_kem_decaps`. Wersje `_internal` niczego nie sprawdzają, bo standard zakłada, że wywołuje się je tylko z poprawnymi danymi.

### 13.6 W kodzie

W `kem.py`:

| Funkcja | Algorytm | Uwagi |
|---|---|---|
| `ml_kem_keygen_internal(params, d, z)` | 16 | bez losowania; służy do testów z ustalonymi ziarnami |
| `ml_kem_encaps_internal(params, ek, m)` | 17 | bez losowania |
| `ml_kem_decaps_internal(params, dk, c)` | 18 | |
| `ml_kem_keygen(params)` | 19 | losuje $d$, $z$ (`secrets.token_bytes`) i woła wersję `_internal` |
| `ml_kem_encaps(params, ek)` | 20 | losuje $m$ |
| `ml_kem_decaps(params, dk, c)` | 21 | |

Wersje `_internal` istnieją po to, żeby w testach podać te same $d$, $z$, $m$ naszemu kodowi i kyber-py i porównać wyniki bajt po bajcie.

Szyfrogramy porównuje się funkcją `secrets.compare_digest`, która działa w czasie niezależnym od miejsca pierwszej różnicy. Zwykłe `==` kończy porównanie na pierwszym różnym bajcie, więc czas jego działania zdradzałby, w którym miejscu szyfrogramy się różnią.

### 13.7 Na co uważać

- Test „$\mathrm{Decaps}(\mathrm{Encaps}(ek)) = K$” przechodzi także wtedy, gdy transformacja FO jest zepsuta albo jej w ogóle nie ma. Dlatego potrzebny jest osobny test: zmieniony bajt w $c$ musi dać dokładnie $J(z \| c)$.
- $\bar{K}$ trzeba liczyć zawsze, a nie tylko po nieudanym porównaniu, żeby czas działania nie zależał od wyniku.

**Źródła:** [FIPS 203 §6 i §7, Alg. 16–21](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf); artykuły o transformacji FO z wcześniejszych notatek (Lukas Prokop, słowniczek FO na nhimg); [CrypTool](https://www.cryptool.org/en/cto/) (zakładki Encap i Decap odpowiadają Alg. 17 i 18).

---

## 14. Rozmiary kluczy i szyfrogramu

Wielomian zakodowany na $d$ bitach na współczynnik zajmuje $256 \cdot d / 8 = 32d$ bajtów. Z tego wynikają wszystkie rozmiary:

- **Klucz publiczny $ek$:** $k$ wielomianów po 12 bitów ($384$ B każdy) plus ziarno $\rho$ (32 B). Razem $384k + 32$.
- **Klucz prywatny $dk$:** $\hat{s}$ ($384k$) + cały $ek$ ($384k + 32$) + $H(ek)$ (32) + $z$ (32). Razem $768k + 96$.
- **Szyfrogram $c$:** $k$ wielomianów po $d_u$ bitów plus jeden po $d_v$ bitów. Razem $32(d_u k + d_v)$.

| Wariant | $ek$ | $dk$ | $c$ | $K$ |
|---|---|---|---|---|
| ML-KEM-512 | 800 B | 1632 B | 768 B | 32 B |
| ML-KEM-768 | 1184 B | 2400 B | 1088 B | 32 B |
| ML-KEM-1024 | 1568 B | 3168 B | 1568 B | 32 B |

Układ klucza prywatnego w bajtach, używany w `ml_kem_decaps_internal`:

| Część | Od | Do |
|---|---|---|
| $dk_{pke}$ | 0 | $384k$ |
| $ek$ | $384k$ | $768k + 32$ |
| $H(ek)$ | $768k + 32$ | $768k + 64$ |
| $z$ | $768k + 64$ | $768k + 96$ |

---

## 15. Ściąga: wzór, algorytm, funkcja w kodzie

| Wzór lub operacja | FIPS 203 | Funkcja | Plik |
|---|---|---|---|
| $H$, $G$, $J$, PRF, XOF | §4.1 | `h_hash`, `g_hash`, `j_hash`, `prf`, `xof` | `crypto_primitives.py` |
| bity ↔ bajty | Alg. 3, 4 | `bits_to_bytes`, `bytes_to_bits` | `compress.py` |
| ByteEncode / ByteDecode | Alg. 5, 6 | `byte_encode`, `byte_decode` | `compress.py` |
| Compress / Decompress | wzory 4.7, 4.8 | `compress`, `decompress` | `compress.py` |
| element macierzy $\hat{A}_{i,j}$ | Alg. 7 | `sample_ntt` | `sampling.py` |
| szum CBD$_\eta$ | Alg. 8 | `sample_poly_cbd` | `sampling.py` |
| $f + g$, $f - g$ | §2.4.5 | `poly_add`, `poly_sub` | `polynomial.py` |
| $f \cdot g$ (wzorzec do testów) | — | `schoolbook_mul` | `polynomial.py` |
| NTT, NTT$^{-1}$ | Alg. 9, 10 | `ntt`, `inv_ntt`, `ntt_vec`, `inv_ntt_vec` | `polynomial.py` |
| $\hat{f} \circ \hat{g}$ | Alg. 11, 12 | `multiply_ntts`, `base_case_multiply` | `polynomial.py` |
| $\hat{A} \circ \hat{s}$, $\hat{t}^\top \circ \hat{y}$, $\hat{A}^\top$ | §2.4.7 | `matvec`, `dot`, `transpose` | `polynomial.py` |
| K-PKE | Alg. 13–15 | `k_pke_keygen`, `k_pke_encrypt`, `k_pke_decrypt` | `pke.py` |
| ML-KEM (bez losowania) | Alg. 16–18 | `ml_kem_*_internal` | `kem.py` |
| ML-KEM | Alg. 19–21 | `ml_kem_keygen`, `ml_kem_encaps`, `ml_kem_decaps` | `kem.py` |

---

## 16. Stan implementacji i ograniczenia

Stan na 18.11.2026 (gałąź `main`, commit `96ec135`, po PR #9):

- **Gotowe:** parametry, funkcje skrótu, kompresja i kodowanie bajtów, oba rodzaje losowania, cała arytmetyka wielomianów z NTT oraz operacjami na wektorach i macierzach, K-PKE na NTT i ML-KEM z transformacją FO i kontrolą danych wejściowych (rozdział 13.5). Wszystkie 311 testów przechodzi.
- **Zgodność ze standardem:** `ek`, `dk`, `c` i `K` są bajt w bajt takie same jak w kyber-py dla trzech wariantów, a wymiana klucza działa w obie strony (nasze klucze z kyber-py i odwrotnie).
- **Brak testu z PyCryptodome**, który był w planie fazy 2 (biblioteki nie ma też w zależnościach).
- **Odporność na ataki czasowe jest ograniczona.** `compare_digest` porównuje szyfrogramy w stałym czasie, ale wybór klucza to zwykły `if`, a Python i NumPy w wielu miejscach działają w czasie zależnym od danych. To implementacja edukacyjna, nieprzeznaczona do zastosowań produkcyjnych.
