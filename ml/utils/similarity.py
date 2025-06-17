"""
Similarity calculation utilities for WhiskeyHub recommendation system.

This module provides efficient similarity calculation functions for both
user-user and item-item collaborative filtering approaches.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.sparse import csr_matrix
import warnings

# Set up logging
logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    Efficient similarity calculator for recommendation systems.
    
    Provides optimized similarity calculations for user-user and item-item
    collaborative filtering, with support for sparse matrices and caching.
    """
    
    def __init__(self, metric: str = 'cosine', min_common_items: int = 2):
        """
        Initialize similarity calculator.
        
        Args:
            metric: Similarity metric ('cosine', 'euclidean', 'pearson')
            min_common_items: Minimum overlapping items required for similarity
        """
        self.metric = metric
        self.min_common_items = min_common_items
        self._similarity_cache: Dict[str, np.ndarray] = {}
        
        logger.info(f"Initialized SimilarityCalculator with {metric} metric")
    
    def calculate_user_similarity(
        self,
        user_item_matrix: pd.DataFrame,
        users: List = None
    ) -> pd.DataFrame:
        """
        Calculate user-user similarity matrix.
        
        Args:
            user_item_matrix: DataFrame with users as rows, items as columns
            users: Optional list of specific users to calculate for
            
        Returns:
            DataFrame with user-user similarities
        """
        logger.info("Calculating user-user similarity matrix...")
        
        # Subset users if specified
        if users is not None:
            available_users = [u for u in users if u in user_item_matrix.index]
            matrix = user_item_matrix.loc[available_users]
        else:
            matrix = user_item_matrix
        
        # Handle missing values
        matrix_filled = matrix.fillna(0)
        
        # Calculate similarity
        if self.metric == 'cosine':
            similarities = self._cosine_similarity_matrix(matrix_filled.values)
        elif self.metric == 'euclidean':
            similarities = self._euclidean_similarity_matrix(matrix_filled.values)
        elif self.metric == 'pearson':
            similarities = self._pearson_similarity_matrix(matrix_filled)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.metric}")
        
        # Convert to DataFrame
        similarity_df = pd.DataFrame(
            similarities,
            index=matrix.index,
            columns=matrix.index
        )
        
        # Apply minimum common items filter
        similarity_df = self._apply_min_common_filter(
            similarity_df, matrix, is_user_similarity=True
        )
        
        logger.info(f"Calculated similarity matrix: {similarity_df.shape}")
        return similarity_df
    
    def calculate_item_similarity(
        self,
        user_item_matrix: pd.DataFrame,
        items: List = None
    ) -> pd.DataFrame:
        """
        Calculate item-item similarity matrix.
        
        Args:
            user_item_matrix: DataFrame with users as rows, items as columns
            items: Optional list of specific items to calculate for
            
        Returns:
            DataFrame with item-item similarities
        """
        logger.info("Calculating item-item similarity matrix...")
        
        # Transpose to get items as rows
        item_user_matrix = user_item_matrix.T
        
        # Subset items if specified
        if items is not None:
            available_items = [i for i in items if i in item_user_matrix.index]
            matrix = item_user_matrix.loc[available_items]
        else:
            matrix = item_user_matrix
        
        # Handle missing values
        matrix_filled = matrix.fillna(0)
        
        # Calculate similarity
        if self.metric == 'cosine':
            similarities = self._cosine_similarity_matrix(matrix_filled.values)
        elif self.metric == 'euclidean':
            similarities = self._euclidean_similarity_matrix(matrix_filled.values)
        elif self.metric == 'pearson':
            similarities = self._pearson_similarity_matrix(matrix_filled)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.metric}")
        
        # Convert to DataFrame
        similarity_df = pd.DataFrame(
            similarities,
            index=matrix.index,
            columns=matrix.index
        )
        
        # Apply minimum common items filter
        similarity_df = self._apply_min_common_filter(
            similarity_df, matrix, is_user_similarity=False
        )
        
        logger.info(f"Calculated item similarity matrix: {similarity_df.shape}")
        return similarity_df
    
    def get_most_similar(
        self,
        similarity_matrix: pd.DataFrame,
        target_id: Union[str, int],
        n_similar: int = 10,
        min_similarity: float = 0.1
    ) -> List[Tuple[Union[str, int], float]]:
        """
        Get most similar users/items for a target.
        
        Args:
            similarity_matrix: Similarity matrix DataFrame
            target_id: Target user/item ID
            n_similar: Number of similar items to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (id, similarity_score) tuples
        """
        if target_id not in similarity_matrix.index:
            logger.warning(f"Target {target_id} not found in similarity matrix")
            return []
        
        # Get similarity scores for target
        similarities = similarity_matrix.loc[target_id]
        
        # Remove self-similarity
        similarities = similarities.drop(target_id, errors='ignore')
        
        # Filter by minimum similarity
        similarities = similarities[similarities >= min_similarity]
        
        # Sort and get top N
        top_similar = similarities.nlargest(n_similar)
        
        return list(zip(top_similar.index, top_similar.values))
    
    def _cosine_similarity_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            similarities = cosine_similarity(matrix)
        
        # Handle NaN values
        similarities = np.nan_to_num(similarities, nan=0.0)
        return similarities
    
    def _euclidean_similarity_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate euclidean similarity matrix (distance converted to similarity)."""
        distances = euclidean_distances(matrix)
        
        # Convert distances to similarities (higher = more similar)
        max_distance = distances.max()
        if max_distance > 0:
            similarities = 1 - (distances / max_distance)
        else:
            similarities = np.ones_like(distances)
        
        return similarities
    
    def _pearson_similarity_matrix(self, matrix: pd.DataFrame) -> np.ndarray:
        """Calculate Pearson correlation similarity matrix."""
        # Calculate correlation matrix
        correlation = matrix.T.corr()
        
        # Handle NaN values
        correlation = correlation.fillna(0)
        
        return correlation.values
    
    def _apply_min_common_filter(
        self,
        similarity_df: pd.DataFrame,
        original_matrix: pd.DataFrame,
        is_user_similarity: bool = True
    ) -> pd.DataFrame:
        """Apply minimum common items/users filter to similarity matrix."""
        if self.min_common_items <= 1:
            return similarity_df
        
        logger.info(f"Applying minimum common items filter: {self.min_common_items}")
        
        # Create mask for valid similarities
        mask = pd.DataFrame(
            True,
            index=similarity_df.index,
            columns=similarity_df.columns
        )
        
        for i, idx1 in enumerate(similarity_df.index):
            for j, idx2 in enumerate(similarity_df.columns):
                if idx1 != idx2:
                    if is_user_similarity:
                        # Count common items rated by both users
                        user1_items = original_matrix.loc[idx1].notna()
                        user2_items = original_matrix.loc[idx2].notna()
                        common_count = (user1_items & user2_items).sum()
                    else:
                        # Count common users who rated both items
                        item1_users = original_matrix[idx1].notna()
                        item2_users = original_matrix[idx2].notna()
                        common_count = (item1_users & item2_users).sum()
                    
                    if common_count < self.min_common_items:
                        mask.iloc[i, j] = False
        
        # Apply mask
        filtered_similarity = similarity_df.where(mask, 0)
        return filtered_similarity


def calculate_user_similarity(
    user_item_matrix: pd.DataFrame,
    metric: str = 'cosine',
    min_common_items: int = 2
) -> pd.DataFrame:
    """
    Convenience function to calculate user-user similarity.
    
    Args:
        user_item_matrix: DataFrame with users as rows, items as columns
        metric: Similarity metric to use
        min_common_items: Minimum overlapping items required
        
    Returns:
        User-user similarity DataFrame
    """
    calculator = SimilarityCalculator(metric=metric, min_common_items=min_common_items)
    return calculator.calculate_user_similarity(user_item_matrix)


def calculate_item_similarity(
    user_item_matrix: pd.DataFrame,
    metric: str = 'cosine',
    min_common_items: int = 2
) -> pd.DataFrame:
    """
    Convenience function to calculate item-item similarity.
    
    Args:
        user_item_matrix: DataFrame with users as rows, items as columns
        metric: Similarity metric to use
        min_common_items: Minimum overlapping users required
        
    Returns:
        Item-item similarity DataFrame
    """
    calculator = SimilarityCalculator(metric=metric, min_common_items=min_common_items)
    return calculator.calculate_item_similarity(user_item_matrix)


def find_neighbors(
    similarity_matrix: pd.DataFrame,
    target_id: Union[str, int],
    n_neighbors: int = 10,
    min_similarity: float = 0.1
) -> List[Tuple[Union[str, int], float]]:
    """
    Find most similar neighbors for a target user/item.
    
    Args:
        similarity_matrix: Similarity matrix DataFrame
        target_id: Target user/item ID
        n_neighbors: Number of neighbors to return
        min_similarity: Minimum similarity threshold
        
    Returns:
        List of (neighbor_id, similarity_score) tuples
    """
    calculator = SimilarityCalculator()
    return calculator.get_most_similar(
        similarity_matrix, target_id, n_neighbors, min_similarity
    )


def sparse_cosine_similarity(
    matrix: Union[np.ndarray, csr_matrix],
    target_vector: Union[np.ndarray, csr_matrix] = None
) -> np.ndarray:
    """
    Calculate cosine similarity for sparse matrices efficiently.
    
    Args:
        matrix: Input matrix (rows are samples)
        target_vector: Optional single vector to compare against all rows
        
    Returns:
        Similarity scores
    """
    if target_vector is not None:
        # Calculate similarity against single vector
        similarities = cosine_similarity(matrix, target_vector.reshape(1, -1))
        return similarities.flatten()
    else:
        # Calculate full similarity matrix
        return cosine_similarity(matrix)


class CachedSimilarityCalculator(SimilarityCalculator):
    """
    Similarity calculator with caching for performance optimization.
    
    Caches computed similarity matrices to avoid recomputation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_enabled = True
        self._cache_hits = 0
        self._cache_misses = 0
    
    def calculate_user_similarity(self, user_item_matrix: pd.DataFrame, users: List = None) -> pd.DataFrame:
        """Calculate user similarity with caching."""
        cache_key = self._get_cache_key(user_item_matrix, users, 'user')
        
        if self._cache_enabled and cache_key in self._similarity_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for user similarity: {cache_key}")
            return self._similarity_cache[cache_key]
        
        self._cache_misses += 1
        result = super().calculate_user_similarity(user_item_matrix, users)
        
        if self._cache_enabled:
            self._similarity_cache[cache_key] = result
            logger.debug(f"Cached user similarity: {cache_key}")
        
        return result
    
    def calculate_item_similarity(self, user_item_matrix: pd.DataFrame, items: List = None) -> pd.DataFrame:
        """Calculate item similarity with caching."""
        cache_key = self._get_cache_key(user_item_matrix, items, 'item')
        
        if self._cache_enabled and cache_key in self._similarity_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for item similarity: {cache_key}")
            return self._similarity_cache[cache_key]
        
        self._cache_misses += 1
        result = super().calculate_item_similarity(user_item_matrix, items)
        
        if self._cache_enabled:
            self._similarity_cache[cache_key] = result
            logger.debug(f"Cached item similarity: {cache_key}")
        
        return result
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_size': len(self._similarity_cache),
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        }
    
    def clear_cache(self) -> None:
        """Clear similarity cache."""
        self._similarity_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Similarity cache cleared")
    
    def _get_cache_key(self, matrix: pd.DataFrame, subset: List, calc_type: str) -> str:
        """Generate cache key for similarity calculation."""
        matrix_hash = hash(tuple(matrix.shape) + tuple(matrix.index) + tuple(matrix.columns))
        subset_hash = hash(tuple(subset)) if subset else 0
        return f"{calc_type}_{self.metric}_{matrix_hash}_{subset_hash}_{self.min_common_items}"