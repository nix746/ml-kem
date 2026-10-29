> Stałe: $q = 3329$, $n = 256$, $k \in {2,3,4}$ (ML-KEM-512 / 768 / 1024). `ndarray` = tablica numpy.

**Skróty źródeł:** [FIPS 203](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf) = standard · [Naskręcki](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) = rozdz. 41 (ML-KEM w NumPy) · [Broeckelmann](https://medium.com/@robert.broeckelmann/post-quantum-cryptography-ml-kem-the-math-7382776f6c44) = intuicja · [Lyubashevsky](https://eprint.iacr.org/2024/1287) i [Menezes](https://eprint.iacr.org/2026/1098) = podstawy krat/LWE · [kyber-py](https://kyber-py.readthedocs.io/en/latest/kyber_py.ml_kem.html) = wyrocznia w testach · [PyCryptodome](https://www.pycryptodome.org/) · [CrypTool](https://www.cryptool.org/en/cto/)

---

## 0. **Tło: kraty, LWE, dlaczego to jest trudne**

- Źródła: [Broeckelmann](https://medium.com/@robert.broeckelmann/post-quantum-cryptography-ml-kem-the-math-7382776f6c44) ("What Is a Lattice?", "The Hard Problem", "Why Noise Matters", "Why Quantum Computers Don't Break It"), [Naskręcki 41.2](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) (Moduled-LWE), [Lyubashevsky](https://eprint.iacr.org/2024/1287), [Menezes](https://eprint.iacr.org/2026/1098) (formalnie, numerów sekcji brak).
- LWE w jednym zdaniu: znamy $A$ i $b = As + e$ ($e$ małe), szukamy $s$. Bez $e$ to zwykły układ równań (eliminacja Gaussa), z $e$ jest trudne.
- ML-KEM używa wersji **Module-LWE**: zamiast liczb mamy wielomiany, a macierz jest mała ($k \times k$ zamiast ok. $700 \times 700$ w zwykłym LWE).
- Trik z ziarnem: $A$ jest losowe, więc nie trzeba go przechowywać ani wysyłać, wystarczy 32-bajtowe $\rho$, z którego $A$ generuje SHAKE. W kluczu publicznym jest więc $\rho$, a nie $A$ ([Lyubashevsky](https://eprint.iacr.org/2024/1287), początek fragmentu o wzorach 13–17).

---

## 1. **Parametry** - [FIPS 203 §8](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf), [kyber-py](https://kyber-py.readthedocs.io/en/latest/kyber_py.ml_kem.html)

### 1.1 Skąd te liczby

- q = 3329 = 13 * 256 + 1, czyli 256 | q-1 (q-1 = 3328 = 13 * 256), istnieje pierwiastek 256-go stopnia z jedynki: $\zeta = 17$.
- Dzięki temu da się zrobić NTT (szybkie mnożenie wielomianów).
- Źródła: [FIPS 203 §2.3 i §4.3](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf) (definicje q, $\zeta$, BitRev7; $\zeta^{128} \equiv -1$), [Naskręcki 41.3 "Primitive Root of Unity"](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html) (sprawdzenie w kodzie).
- Sprawdzian w kodzie: `pow(17, 256, 3329) == 1` oraz `pow(17, 128, 3329) == 3328` (czyli $-1$).

### 1.2 Zestawy parametrów

|Wariant|$k$|$\eta_1$|$\eta_2$|$d_u$|$d_v$|
|---|---|---|---|---|---|
|512|2|3|2|10|4|
|768|3|2|2|10|4|
|1024|4|2|2|11|5|

- **W kodzie:** zwykłe stałe + `dataclass` / `NamedTuple` na zestaw.
- Źródła: [FIPS 203 §8, Tabela 2](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf), kyber-py `DEFAULT_PARAMETERS` (klucze `k`, `du`, `dv`, `eta_1`, `eta_2`), [Naskręcki 41.10](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html).

#### 1.3 Co znaczą parametry

- $k$: liczba wielomianów w wektorze (s, e, y) i rozmiar macierzy A (k × k). Im większe $k$, tym wyższy poziom bezpieczeństwa i większe klucze.
- $\eta_1$: rozpiętość szumu CBD dla s, e (w KeyGen) i y (w Encrypt). Współczynniki są z $[-\eta_1, \eta_1]$.
- $\eta_2$: to samo dla e_1 i e_2 w Encrypt.
- $d_u$: na ilu bitach kompresujemy każdy współczynnik wektora u (zamiast 12).
- $d_v$: to samo dla wielomianu v.

Dodatkowo:

- Dla 512 szum $\eta_1 = 3$ jest większy niż w pozostałych wariantach. Tak dobrano parametry w Tabeli 2; zwykle tłumaczy się to kompromisem bezpieczeństwo / błąd odszyfrowania (przy mniejszym $k$ jest mniej "wymiarów"). [FIPS 203 §3.2](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf).
- Poziomy bezpieczeństwa NIST: 512 ≈ kategoria 1 (jak AES-128), 768 ≈ kategoria 3 (AES-192), 1024 ≈ kategoria 5 (AES-256).
- $d_u$ i $d_v$ wpływają na rozmiar szyfrogramu: $|c| = 32(d_u \cdot k + d_v)$. Są dobrane tak, żeby błąd kompresji nie psuł odszyfrowania.
- Prawdopodobieństwo błędu odszyfrowania: [FIPS 203 §3.2, Tabela 1](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf). Doświadczalnie: [Naskręcki 41.12](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html).

---

## 2. **Pierścień wielomianów $R_q$** - [FIPS 203 §2.4.4–2.4.5](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf), [Naskręcki 41.2](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html)

### 2.1 Matematyka

- R_q = Z_q[X]/(X^{256}+1)
- Element: f = a_0 + a_1 * X + ... + a_{255} X^255, współczynniki mod 3329.
- Mnożenie jest modulo $X^{256}+1$, czyli $X^{256} = -1$. Jak wyjdziemy poza $X^{255}$, wynik zawija się ze zmianą znaku.
- Przykład (zawijanie): $X^{3} \cdot X^{2} = X^{5}$, a w pierścieniu z $n = 4$ ($X^4 = -1$) to $X^5 = X \cdot X^4 = -X$.
- Intuicja: to wielomiany "po okręgu", ale z przeskokiem znaku. Nazywa się to mnożeniem **negacyklicznym**.
- Dla czytelnego opisu bez nazywania $R_q$: [Broeckelmann "Polynomial Arithmetic"](https://medium.com/@robert.broeckelmann/post-quantum-cryptography-ml-kem-the-math-7382776f6c44).

### 2.2 W kodzie

- `np.ndarray`, shape `(256,)`, indeks = potęga X.
- Dodawanie: `(a + b) % q`.
- Typ: `int64` (w `int32` sumy mogą się przepełnić).
- Operacje pomocnicze (dodawanie, odejmowanie, iloczyn skalarny wektorów): [Naskręcki 41.6](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html).

### 2.3 Alternatywy

- Lista Pythona: działa, ale wolna i bez operacji wektorowych.
- Klasa `Poly` opakowująca tablicę: ładniejszy kod (`a + b`), więcej pisania.

### 2.4 Pułapka

- Mnożenie zwykłe daje 256^2 operacji + zawijanie ze znakiem. Dlatego stosujemy NTT.
- Pomiar: [Naskręcki 41.4 "Timing Comparison: Schoolbook vs NTT"](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html).

## 3. **Wektory i macierze** - [FIPS 203 §2.4.6–2.4.7](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf), [Naskręcki 41.14](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html)

### 3.1 Matematyka

- s, e $\in$ R_q^k (wektor $k$ wielomianów), A$\in$ R_q^(k * k) .
- As_i = $\sum_j$ A_ij $\circ$ s_j, gdzie "$\circ$" to mnożenie wielomianów, nie liczb.
- Wersja bez wielomianów, ale z macierzami i szumem: [Broeckelmann "Matrix Multiplication with Polynomials"](https://medium.com/@robert.broeckelmann/post-quantum-cryptography-ml-kem-the-math-7382776f6c44).

### 3.2 W kodzie

|Obiekt|shape|
|---|---|
|wielomian|`(256,)`|
|wektor|`(k, 256)`|
|macierz|`(k, k, 256)`|

- Ostatni wymiar to zawsze wielomian, poprzednie to indeksy macierzowe.

### 3.3 Pułapka

- `*` i `@` z numpy nie robią mnożenia w R_q. Potrzebna własna funkcja + suma po j mod q.
- Szkic mnożenia macierz × wektor w dziedzinie NTT: dla każdego `i` zsumuj po `j` wyniki `multiply_ntts(A[i, j], s[j])` i zredukuj mod q.

---

## 4. **NTT** - [FIPS 203 §4.3](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf) (Alg. 9, 10), [Naskręcki 41.3–41.4](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html)

### 4.1 Matematyka

- Idea (intuicja): zamiast mnożyć wielomiany współczynnik po współczynniku, liczymy je w wielu punktach (pierwiastkach z jedynki), mnożymy wartości punkt po punkcie i wracamy. Dobrze opisuje to seria [RareSkills: Number Theoretic Transform](https://rareskills.io/post/number-theoretic-transform) (pierwiastki z jedynki, odwrotne NTT, mnożenie wielomianów, kod w Pythonie).
- X^256 + 1 rozkłada się na 128 czynników X^2 - $\gamma_i$, gdzie $\gamma_i = \zeta^{2 BitRev_7(i) + 1}$.
- NTT(f) = reszty f z dzielenia przez każdy z tych czynników, czyli 128 wielomianów stopnia $\le$ 1 (128 par współczynników).
- Algorytm to motylki w warstwach len = 128, 64, ..., 2:
    - t = $\zeta^{...}$ * f[j+len]
    - f[j+len] = f[j] - t
    - f[j] = f[j] + t
    - wszystko mod q
- InvNTT: to samo w drugą stronę + na końcu mnożenie przez 128^{-1} = 3303 mod q (sprawdzenie: 128 * 3303 = 3329 * 127 + 1).
- Wizualnie: [Naskręcki 41.4 "NTT Butterfly Diagram"](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html).
- **Uwaga o źródle RareSkills:** pokazuje standardowe (pełne) NTT na małym polu (q = 17), bez motylków w stylu Kybera, bez mnożenia negacyklicznego i bez wariantu "kończy się na parach". Do samego ML-KEM używaj FIPS 203 §4.3 i Naskręckiego.

### 4.2 W [kodzie](https://rareskills.io/post/number-theoretic-transform)

- Dalej `(256,)`, ale znaczenie inne: 128 par (f[2i], f[2i+1]).
- Tablica $\zeta$: 128 potęg 17 w kolejności bit-reversed, nie rosnącej. Gotowa tablica do porównania: [FIPS 203, Dodatek A](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf).
- Na początek zwykłe pętle, łatwiej pokazać co się dzieje.
- Kod referencyjny w NumPy: [Naskręcki 41.4](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html), wizualizacja czynników: [41.13 "Visualising Twiddle Factors"](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html).

### 4.3 Alternatywy

- Mnożenie zwykłe: prostsze i wolniejsze, dobre jako wyrocznia do testów NTT. Test: `inv_ntt(ntt(f)) == f` oraz `inv_ntt(multiply_ntts(ntt(f), ntt(g))) == schoolbook(f, g)`.
- Tablicę $\zeta$ policzyć przy imporcie albo wpisać gotową i porównać z FIPS 203.

### 4.4 Pułapka

- NTT kończy się na parach, nie na pojedynczych liczbach, bo 3329 nie jest $\equiv$ 1 mod 512 (3328 = 6,5 * 512). Stąd specjalne mnożenie (pkt 5).

---

## 5. **Mnożenie w dziedzinie NTT** - [FIPS 203 §4.3.1](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf) (Alg. 11, 12), [Naskręcki 41.3 "Pointwise Multiplication"](https://bnaskrecki.faculty.wmi.amu.edu.pl/crypto/book/part14_lattice_crypto/ch41_ml_kem_kyber.html)

### 5.1 Matematyka

- Dla każdej pary: (a_0 + a_1 X)(b_0 + b_1 X) mod (X^2 - $\gamma$)
- c_0 = a_0 b_0 + a_1 b_1 $\gamma$
- c_1 = a_0 b_1 + a_1 b_0
- Wynika to wprost z mnożenia i zamiany $X^2 = \gamma$.

### 5.2 W kodzie

- Pętla po 128 parach albo slicing `a[0::2]`, `a[1::2]` + tablica $\gamma$.

### 5.3 Pułapka

- $\gamma$ jest inna dla każdej pary.