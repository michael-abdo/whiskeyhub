"""
Collaborative filtering model for WhiskeyHub recommendation system.

This model leverages WhiskeyHub's exceptional 58% data density to provide
user-based and item-based collaborative filtering recommendations.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .base_model import BaseCollaborativeModel
from ..utils.similarity import SimilarityCalculator, CachedSimilarityCalculator

# Set up logging
logger = logging.getLogger(__name__)


class CollaborativeRecommender(BaseCollaborativeModel):
    """
    Collaborative filtering recommender using user-user and item-item approaches.
    
    Leverages WhiskeyHub's exceptional 58% data density to find user neighborhoods
    and generate high-quality recommendations based on similar users' preferences.
    
    Key features:
    - User-based collaborative filtering (primary approach)
    - Item-based collaborative filtering (backup)
    - Cold start handling for new users
    - Neighborhood-based prediction algorithms
    - Efficient similarity calculations with caching
    """
    
    def __init__(
        self,
        approach: str = 'user_based',
        similarity_metric: str = 'cosine',
        n_neighbors: int = 20,
        min_similarity: float = 0.1,
        min_common_items: int = 3,
        use_cache: bool = True,
        **kwargs
    ):
        """
        Initialize Collaborative Recommender.
        
        Args:
            approach: 'user_based', 'item_based', or 'hybrid'
            similarity_metric: 'cosine', 'euclidean', or 'pearson'
            n_neighbors: Number of neighbors to consider
            min_similarity: Minimum similarity threshold
            min_common_items: Minimum common items for similarity calculation
            use_cache: Whether to use similarity caching
            **kwargs: Additional arguments passed to BaseCollaborativeModel
        """
        super().__init__(
            model_name='CollaborativeRecommender',
            **kwargs
        )
        
        self.approach = approach
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity
        self.min_common_items = min_common_items
        
        # Initialize similarity calculator
        if use_cache:
            self.similarity_calculator = CachedSimilarityCalculator(
                metric=similarity_metric,
                min_common_items=min_common_items
            )
        else:
            self.similarity_calculator = SimilarityCalculator(
                metric=similarity_metric,
                min_common_items=min_common_items
            )
        
        # Model data
        self.global_mean: float = 0.0
        self.user_means: Dict[Any, float] = {}
        self.item_means: Dict[Any, float] = {}
        
        logger.info(f"Initialized CollaborativeRecommender with {approach} approach")
    
    def fit(self, data: pd.DataFrame, target_col: str = 'rating') -> 'CollaborativeRecommender':
        """
        Train the collaborative filtering model.
        
        Args:
            data: Training dataset with user-item interactions
            target_col: Name of the rating column
            
        Returns:
            Self for method chaining
        """
        logger.info("Training CollaborativeRecommender...")
        
        # Create user-item matrix
        self.user_item_matrix = self._create_user_item_matrix(data)
        logger.info(f"Created user-item matrix: {self.user_item_matrix.shape}")
        
        # Calculate global statistics
        self.global_mean = self.user_item_matrix.stack().mean()
        
        # Calculate user and item means
        self.user_means = self.user_item_matrix.mean(axis=1).to_dict()
        self.item_means = self.user_item_matrix.mean(axis=0).to_dict()
        
        # Calculate similarity matrices based on approach
        if self.approach in ['user_based', 'hybrid']:
            logger.info("Calculating user similarity matrix...")
            self.user_similarities = self.similarity_calculator.calculate_user_similarity(
                self.user_item_matrix
            )
            logger.info(f"User similarity matrix: {self.user_similarities.shape}")
        
        if self.approach in ['item_based', 'hybrid']:
            logger.info("Calculating item similarity matrix...")
            self.item_similarities = self.similarity_calculator.calculate_item_similarity(
                self.user_item_matrix
            )
            logger.info(f"Item similarity matrix: {self.item_similarities.shape}")
        
        # Store training statistics
        data_density = self.user_item_matrix.notna().sum().sum() / (
            self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        )
        
        self.training_stats = {
            'n_users': self.user_item_matrix.shape[0],
            'n_items': self.user_item_matrix.shape[1],
            'n_ratings': self.user_item_matrix.notna().sum().sum(),
            'data_density': data_density,
            'global_mean': self.global_mean,
            'approach': self.approach,
            'similarity_metric': self.similarity_metric,
        }
        
        logger.info(f"Training completed. Data density: {data_density:.3f}")
        
        self.is_fitted = True
        return self
    
    def predict_rating(
        self,
        user_id: Any,
        whiskey_id: Any,
        user_features: Dict[str, Any] = None,
        whiskey_features: Dict[str, Any] = None
    ) -> float:
        """
        Predict rating for a user-item pair using collaborative filtering.
        
        Args:
            user_id: User identifier
            whiskey_id: Whiskey identifier
            user_features: Optional user features (not used)
            whiskey_features: Optional whiskey features (not used)
            
        Returns:
            Predicted rating
        """
        self._validate_fitted()
        
        # Handle cold start cases
        if user_id not in self.user_item_matrix.index:
            return self._handle_cold_start_user(whiskey_id)
        
        if whiskey_id not in self.user_item_matrix.columns:
            return self._handle_cold_start_item(user_id)
        
        # Check if user has already rated this item
        existing_rating = self.user_item_matrix.loc[user_id, whiskey_id]
        if not pd.isna(existing_rating):
            return existing_rating
        
        # Predict using selected approach
        if self.approach == 'user_based':
            prediction = self._predict_user_based(user_id, whiskey_id)
        elif self.approach == 'item_based':
            prediction = self._predict_item_based(user_id, whiskey_id)
        elif self.approach == 'hybrid':
            user_pred = self._predict_user_based(user_id, whiskey_id)
            item_pred = self._predict_item_based(user_id, whiskey_id)
            # Simple average of both approaches
            prediction = (user_pred + item_pred) / 2
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
        
        return np.clip(prediction, 0, 10)
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_rated: bool = True,
        min_rating: float = None
    ) -> List[Tuple[Any, float]]:
        """
        Generate personalized recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude already-rated items
            min_rating: Minimum predicted rating threshold
            
        Returns:
            List of (whiskey_id, predicted_rating) tuples
        """
        self._validate_fitted()
        
        # Handle cold start users
        if user_id not in self.user_item_matrix.index:
            return self._recommend_cold_start_user(n_recommendations, min_rating)
        
        recommendations = []
        
        # Get all items
        all_items = self.user_item_matrix.columns
        
        # Filter items if needed
        if exclude_rated:
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_items = user_ratings[user_ratings.isna()].index
        else:
            unrated_items = all_items
        
        # Predict ratings for unrated items
        for item_id in unrated_items:
            try:
                predicted_rating = self.predict_rating(user_id, item_id)
                
                # Apply minimum rating filter
                if min_rating is None or predicted_rating >= min_rating:
                    recommendations.append((item_id, predicted_rating))
                    
            except Exception as e:
                logger.warning(f"Failed to predict for item {item_id}: {e}")
        
        # Sort by predicted rating and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def _predict_user_based(self, user_id: Any, item_id: Any) -> float:
        """Predict rating using user-based collaborative filtering."""
        if self.user_similarities is None:
            raise ValueError("User similarities not calculated")
        
        # Find similar users who rated this item
        similar_users = self.similarity_calculator.get_most_similar(
            self.user_similarities,
            user_id,
            n_similar=self.n_neighbors,
            min_similarity=self.min_similarity
        )
        
        # Filter users who rated this item
        valid_neighbors = []
        for neighbor_id, similarity in similar_users:
            if neighbor_id in self.user_item_matrix.index:
                neighbor_rating = self.user_item_matrix.loc[neighbor_id, item_id]
                if not pd.isna(neighbor_rating):
                    valid_neighbors.append((neighbor_id, similarity, neighbor_rating))
        
        if not valid_neighbors:
            # Fallback to user mean or global mean
            return self.user_means.get(user_id, self.global_mean)
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        user_mean = self.user_means.get(user_id, self.global_mean)
        
        for neighbor_id, similarity, rating in valid_neighbors:
            neighbor_mean = self.user_means.get(neighbor_id, self.global_mean)
            # Mean-centered rating
            rating_deviation = rating - neighbor_mean
            
            numerator += similarity * rating_deviation
            denominator += abs(similarity)
        
        if denominator > 0:
            prediction = user_mean + (numerator / denominator)
        else:
            prediction = user_mean
        
        return prediction
    
    def _predict_item_based(self, user_id: Any, item_id: Any) -> float:
        """Predict rating using item-based collaborative filtering."""
        if self.item_similarities is None:
            raise ValueError("Item similarities not calculated")
        
        # Find similar items rated by this user
        similar_items = self.similarity_calculator.get_most_similar(
            self.item_similarities,
            item_id,
            n_similar=self.n_neighbors,
            min_similarity=self.min_similarity
        )
        
        # Filter items rated by this user
        valid_neighbors = []
        for neighbor_item, similarity in similar_items:
            if neighbor_item in self.user_item_matrix.columns:
                user_rating = self.user_item_matrix.loc[user_id, neighbor_item]
                if not pd.isna(user_rating):
                    valid_neighbors.append((neighbor_item, similarity, user_rating))
        
        if not valid_neighbors:
            # Fallback to item mean or global mean
            return self.item_means.get(item_id, self.global_mean)
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for neighbor_item, similarity, rating in valid_neighbors:
            numerator += similarity * rating
            denominator += abs(similarity)
        
        if denominator > 0:
            prediction = numerator / denominator
        else:
            prediction = self.item_means.get(item_id, self.global_mean)
        
        return prediction
    
    def _handle_cold_start_user(self, item_id: Any) -> float:
        """Handle prediction for new users."""
        # Return item mean or global mean
        return self.item_means.get(item_id, self.global_mean)
    
    def _handle_cold_start_item(self, user_id: Any) -> float:
        """Handle prediction for new items."""
        # Return user mean or global mean
        return self.user_means.get(user_id, self.global_mean)
    
    def _recommend_cold_start_user(
        self,
        n_recommendations: int,
        min_rating: float = None
    ) -> List[Tuple[Any, float]]:
        """Generate recommendations for cold start users."""
        # Recommend most popular items (highest mean ratings)
        item_ratings = []
        
        for item_id in self.user_item_matrix.columns:
            item_mean = self.item_means.get(item_id, self.global_mean)
            
            if min_rating is None or item_mean >= min_rating:
                item_ratings.append((item_id, item_mean))
        
        # Sort by rating and return top N
        item_ratings.sort(key=lambda x: x[1], reverse=True)
        return item_ratings[:n_recommendations]
    
    def get_user_neighborhood(self, user_id: Any) -> List[Tuple[Any, float]]:
        """
        Get the neighborhood (similar users) for a given user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of (similar_user_id, similarity_score) tuples
        """
        self._validate_fitted()
        
        if self.user_similarities is None:
            raise ValueError("User similarities not calculated")
        
        return self.similarity_calculator.get_most_similar(
            self.user_similarities,
            user_id,
            n_similar=self.n_neighbors,
            min_similarity=self.min_similarity
        )
    
    def get_item_neighborhood(self, item_id: Any) -> List[Tuple[Any, float]]:
        """
        Get the neighborhood (similar items) for a given item.
        
        Args:
            item_id: Item identifier
            
        Returns:
            List of (similar_item_id, similarity_score) tuples
        """
        self._validate_fitted()
        
        if self.item_similarities is None:
            raise ValueError("Item similarities not calculated")
        
        return self.similarity_calculator.get_most_similar(
            self.item_similarities,
            item_id,
            n_similar=self.n_neighbors,
            min_similarity=self.min_similarity
        )
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Get model-specific data for saving."""
        return {
            'approach': self.approach,
            'similarity_metric': self.similarity_metric,
            'n_neighbors': self.n_neighbors,
            'min_similarity': self.min_similarity,
            'min_common_items': self.min_common_items,
            'user_item_matrix': self.user_item_matrix,
            'user_similarities': getattr(self, 'user_similarities', None),
            'item_similarities': getattr(self, 'item_similarities', None),
            'global_mean': self.global_mean,
            'user_means': self.user_means,
            'item_means': self.item_means,
        }
    
    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """Load model-specific data from saved state."""
        self.approach = data['approach']
        self.similarity_metric = data['similarity_metric']
        self.n_neighbors = data['n_neighbors']
        self.min_similarity = data['min_similarity']
        self.min_common_items = data['min_common_items']
        self.user_item_matrix = data['user_item_matrix']
        self.user_similarities = data.get('user_similarities')
        self.item_similarities = data.get('item_similarities')
        self.global_mean = data['global_mean']
        self.user_means = data['user_means']
        self.item_means = data['item_means']
        
        # Reinitialize similarity calculator
        self.similarity_calculator = SimilarityCalculator(
            metric=self.similarity_metric,
            min_common_items=self.min_common_items
        )


# Convenience function
def train_collaborative_recommender(
    data: pd.DataFrame,
    approach: str = 'user_based',
    **kwargs
) -> CollaborativeRecommender:
    """
    Convenience function to train a collaborative recommender.
    
    Args:
        data: Training dataset
        approach: Collaborative filtering approach
        **kwargs: Additional arguments for CollaborativeRecommender
        
    Returns:
        Trained CollaborativeRecommender
    """
    model = CollaborativeRecommender(approach=approach, **kwargs)
    return model.fit(data)