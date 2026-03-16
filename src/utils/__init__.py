"""
Utility functions
"""

from .similarity import cosine_similarity_sparse, pearson_correlation
from .preprocessing import normalize_ratings, filter_sparse_data

__all__ = [
    'cosine_similarity_sparse',
    'pearson_correlation',
    'normalize_ratings',
    'filter_sparse_data'
]

