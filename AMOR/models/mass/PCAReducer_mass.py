"""Shim — mass PCA lives in models/PCAReducer.py."""
from ..PCAReducer import (
    PCAReducerMultiChannelMass,
    _MassPCAView,
    pca_like_from_rows_mass,
    fit_random_init_basis_mass,
)
__all__ = [
    'PCAReducerMultiChannelMass',
    '_MassPCAView',
    'pca_like_from_rows_mass',
    'fit_random_init_basis_mass',
]
