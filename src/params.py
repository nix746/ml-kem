"""ML-KEM parameters as specified in NIST FIPS 203 (Section 8, Table 2)."""

from dataclasses import dataclass

N = 256
Q = 3329
ZETA = 17
N_INV = pow(128, -1, Q)

assert 128 * N_INV % Q == 1


@dataclass(frozen=True)
class MLKEMParams:
    """Parameter set for a specific ML-KEM security level."""

    name: str
    k: int
    eta1: int
    eta2: int
    du: int
    dv: int

    @property
    def ek_size(self) -> int:
        """Public encapsulation key size in bytes."""
        return 384 * self.k + 32

    @property
    def dk_size(self) -> int:
        """Decapsulation key size in bytes."""
        return 768 * self.k + 96

    @property
    def ct_size(self) -> int:
        """Ciphertext size in bytes."""
        return 32 * (self.du * self.k + self.dv)


ML_KEM_512 = MLKEMParams("ML-KEM-512", k=2, eta1=3, eta2=2, du=10, dv=4)
ML_KEM_768 = MLKEMParams("ML-KEM-768", k=3, eta1=2, eta2=2, du=10, dv=4)
ML_KEM_1024 = MLKEMParams("ML-KEM-1024", k=4, eta1=2, eta2=2, du=11, dv=5)
