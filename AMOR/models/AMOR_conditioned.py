"""AMOR variants with input-field and FiLM conditioning.

Both are thin aliases of the unified ``AMOR`` (models/amor.py): FiLM is a
constructor option there, and mass weighting follows the PCA reducer.
"""

from __future__ import annotations

from .amor import AMOR

AMORWithFieldInputs = AMOR
AMORWithLoadFiLM = AMOR

__all__ = ["AMORWithFieldInputs", "AMORWithLoadFiLM"]
