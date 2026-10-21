"""Parametry algorytmu ML-KEM zgodnie ze specyfikacją NIST FIPS 203 (Sekcja 8, Tabela 2)."""

from dataclasses import dataclass

# TODO: Zdefiniować stałe pierścienia (N, Q) oraz zestawy parametrów dla ML-KEM-512, 768, 1024


@dataclass(frozen=True)
class MLKEMParams:
    """Zestaw parametrów dla konkretnego wariantu bezpieczeństwa ML-KEM."""

    pass
