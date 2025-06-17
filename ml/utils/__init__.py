"""
Utility functions module for WhiskeyHub ML pipeline.

This module contains:
- Similarity calculation functions (user-user, item-item)
- Sparse matrix operations
- Caching utilities
- Data transformation helpers
- Common algorithms and helper functions
"""

from .similarity import (
    SimilarityCalculator, 
    CachedSimilarityCalculator,
    calculate_user_similarity, 
    calculate_item_similarity,
    find_neighbors,
    sparse_cosine_similarity
)

__all__ = [
    "SimilarityCalculator",
    "CachedSimilarityCalculator", 
    "calculate_user_similarity",
    "calculate_item_similarity",
    "find_neighbors", 
    "sparse_cosine_similarity",
]