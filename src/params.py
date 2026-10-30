"""ML-KEM parameters as specified in NIST FIPS 203 (Section 8, Table 2)."""

from dataclasses import dataclass

N = 256
Q = 3329

# TODO: Define the ring constants (N, Q) and parameter sets for ML-KEM-512, 768, 1024


@dataclass(frozen=True)
class MLKEMParams:
    """Parameter set for a specific ML-KEM security level."""

    pass
