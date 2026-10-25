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