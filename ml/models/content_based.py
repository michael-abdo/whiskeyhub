"""
Content-based filtering model for WhiskeyHub recommendation system.

This model recommends whiskeys based on similarity of whiskey features,
using the proven features from the linear baseline analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

from .base_model import BasePredictiveModel

# Set up logging
logger = logging.getLogger(__name__)


class ContentBasedRecommender(BasePredictiveModel):
    """
    Content-based recommendation model using whiskey feature similarity.
    
    This model recommends whiskeys based on similarity of whiskey characteristics,
    using proven features: complexity, finish_duration, age, proof, price, etc.
    
    Key features:
    - Uses cosine and euclidean similarity metrics
    - Handles missing features gracefully
    - Provides recommendation explanations
    - Supports both feature-based and user-preference-based recommendations
    """
    
    def __init__(
        self,
        features: List[str] = None,
        similarity_metric: str = 'cosine',
        min_similarity: float = 0.1,
        **kwargs
    ):
        """
        Initialize Content-Based Recommender.
        
        Args:
            features: List of whiskey features to use for similarity
            similarity_metric: 'cosine' or 'euclidean'
            min_similarity: Minimum similarity threshold for recommendations
            **kwargs: Additional arguments passed to BasePredictiveModel
        """
        # Use proven features from linear model analysis
        default_features = [
            'complexity', 'finish_duration', 'age', 'proof', 'price', 'profiness', 'viscocity'
        ]
        
        super().__init__(
            features=features or default_features,
            model_name='ContentBasedRecommender',
            **kwargs
        )
        
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity
        
        # Model components
        self.whiskey_features: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.whiskey_ids: List[Any] = []
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized ContentBasedRecommender with {len(self.features)} features")
    
    def fit(self, data: pd.DataFrame, target_col: str = 'rating') -> 'ContentBasedRecommender':
        """
        Train the content-based model by building whiskey feature matrix.
        
        Args:
            data: Training dataset with whiskey features
            target_col: Target column (for compatibility, not used in training)
            
        Returns:
            Self for method chaining
        """
        logger.info("Training ContentBasedRecommender...")
        
        # Find available features
        available_features = [f for f in self.features if f in data.columns]
        if not available_features:
            raise ValueError("No valid features found for content-based filtering")
        
        self.feature_names = available_features
        logger.info(f"Using features: {self.feature_names}")
        
        # Get unique whiskeys with features
        if 'whiskey_id' in data.columns:
            # Aggregate features by whiskey (take first non-null value)
            whiskey_data = data.groupby('whiskey_id')[self.feature_names].first().reset_index()
        else:
            # Use data as-is if no whiskey_id
            whiskey_data = data[['whiskey_id'] + self.feature_names].drop_duplicates() if 'whiskey_id' in data.columns else data[self.feature_names].reset_index()
            whiskey_data['whiskey_id'] = whiskey_data.index if 'whiskey_id' not in whiskey_data.columns else whiskey_data['whiskey_id']
        
        # Handle missing values
        whiskey_data_clean = whiskey_data.dropna(subset=self.feature_names)
        logger.info(f"Clean whiskey data: {len(whiskey_data_clean)} whiskeys with complete features")
        
        if len(whiskey_data_clean) < 2:
            raise ValueError("Insufficient whiskey data for similarity calculations")
        
        # Store whiskey data
        self.whiskey_features = whiskey_data_clean.copy()
        self.whiskey_ids = whiskey_data_clean['whiskey_id'].tolist()
        
        # Scale features for similarity calculations
        feature_matrix = whiskey_data_clean[self.feature_names].values
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Calculate similarity matrix
        if self.similarity_metric == 'cosine':
            self.similarity_matrix = cosine_similarity(scaled_features)
        elif self.similarity_metric == 'euclidean':
            # Convert distances to similarities (higher = more similar)
            distances = euclidean_distances(scaled_features)
            max_distance = distances.max()
            self.similarity_matrix = 1 - (distances / max_distance)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        # Calculate feature statistics
        self._calculate_feature_stats(whiskey_data_clean[self.feature_names])
        
        # Store training statistics
        self.training_stats = {
            'n_whiskeys': len(whiskey_data_clean),
            'n_features': len(self.feature_names),
            'similarity_metric': self.similarity_metric,
            'mean_similarity': self.similarity_matrix.mean(),
            'std_similarity': self.similarity_matrix.std(),
        }
        
        self.is_fitted = True
        logger.info("ContentBasedRecommender training completed")
        
        return self
    
    def predict_rating(
        self,
        user_id: Any,
        whiskey_id: Any,
        user_features: Dict[str, Any] = None,
        whiskey_features: Dict[str, Any] = None
    ) -> float:
        """
        Predict rating using content-based approach.
        
        Uses similarity to highly-rated whiskeys to predict rating.
        
        Args:
            user_id: User identifier
            whiskey_id: Whiskey identifier
            user_features: Optional user preference features
            whiskey_features: Optional whiskey features
            
        Returns:
            Predicted rating
        """
        self._validate_fitted()
        
        # Get similar whiskeys
        similar_whiskeys = self.get_similar_whiskeys(whiskey_id, n_similar=10)
        
        if not similar_whiskeys:
            # Fallback to global average or feature-based prediction
            return 7.0  # Average rating from analysis
        
        # Weight predictions by similarity
        weighted_sum = 0
        total_weight = 0
        
        for similar_id, similarity in similar_whiskeys:
            # Use similarity as weight and assume good whiskeys get high ratings
            # This is a simplified approach - in practice you'd use actual ratings
            base_rating = 7.0  # Average baseline
            feature_bonus = self._calculate_feature_bonus(similar_id)
            predicted_rating = base_rating + feature_bonus
            
            weighted_sum += predicted_rating * similarity
            total_weight += similarity
        
        if total_weight > 0:
            prediction = weighted_sum / total_weight
        else:
            prediction = 7.0
        
        return np.clip(prediction, 0, 10)
    
    def recommend(
        self,
        user_id: Any = None,
        target_whiskey_id: Any = None,
        user_preferences: Dict[str, float] = None,
        n_recommendations: int = 10,
        exclude_rated: bool = True,
        min_rating: float = None
    ) -> List[Tuple[Any, float]]:
        """
        Generate content-based recommendations.
        
        Args:
            user_id: User identifier (for compatibility)
            target_whiskey_id: Whiskey to find similar items for
            user_preferences: User preference profile for features
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude rated whiskeys
            min_rating: Minimum rating threshold
            
        Returns:
            List of (whiskey_id, similarity_score) tuples
        """
        self._validate_fitted()
        
        recommendations = []
        
        # Case 1: Recommend based on similar whiskey
        if target_whiskey_id is not None:
            similar_whiskeys = self.get_similar_whiskeys(
                target_whiskey_id, 
                n_similar=n_recommendations + 1  # +1 to exclude self
            )
            # Remove the target whiskey itself
            recommendations = [(wid, sim) for wid, sim in similar_whiskeys if wid != target_whiskey_id]
        
        # Case 2: Recommend based on user preferences
        elif user_preferences is not None:
            recommendations = self._recommend_by_preferences(user_preferences, n_recommendations)
        
        # Case 3: Recommend top-rated whiskeys by features
        else:
            # Use feature-based scoring (complexity and finish_duration are key)
            recommendations = self._recommend_by_feature_quality(n_recommendations)
        
        # Apply minimum rating filter if specified
        if min_rating is not None:
            recommendations = [(wid, score) for wid, score in recommendations if score >= min_rating]
        
        return recommendations[:n_recommendations]
    
    def get_similar_whiskeys(
        self,
        whiskey_id: Any,
        n_similar: int = 10
    ) -> List[Tuple[Any, float]]:
        """
        Find whiskeys similar to the given whiskey.
        
        Args:
            whiskey_id: Target whiskey identifier
            n_similar: Number of similar whiskeys to return
            
        Returns:
            List of (whiskey_id, similarity_score) tuples
        """
        self._validate_fitted()
        
        if whiskey_id not in self.whiskey_ids:
            logger.warning(f"Whiskey {whiskey_id} not found in training data")
            return []
        
        # Get whiskey index
        whiskey_idx = self.whiskey_ids.index(whiskey_id)
        
        # Get similarity scores
        similarities = self.similarity_matrix[whiskey_idx]
        
        # Get indices of most similar whiskeys
        similar_indices = np.argsort(similarities)[::-1]  # Descending order
        
        # Build result list
        similar_whiskeys = []
        for idx in similar_indices[:n_similar]:
            if similarities[idx] >= self.min_similarity:
                similar_whiskeys.append((self.whiskey_ids[idx], similarities[idx]))
        
        return similar_whiskeys
    
    def get_feature_explanation(
        self,
        whiskey_id: Any,
        similar_whiskey_id: Any
    ) -> Dict[str, float]:
        """
        Explain why two whiskeys are similar based on features.
        
        Args:
            whiskey_id: First whiskey
            similar_whiskey_id: Second whiskey
            
        Returns:
            Dictionary with feature similarity explanations
        """
        self._validate_fitted()
        
        if whiskey_id not in self.whiskey_ids or similar_whiskey_id not in self.whiskey_ids:
            return {}
        
        # Get whiskey features
        whiskey_row = self.whiskey_features[self.whiskey_features['whiskey_id'] == whiskey_id].iloc[0]
        similar_row = self.whiskey_features[self.whiskey_features['whiskey_id'] == similar_whiskey_id].iloc[0]
        
        explanations = {}
        for feature in self.feature_names:
            val1 = whiskey_row[feature]
            val2 = similar_row[feature]
            
            if pd.notna(val1) and pd.notna(val2):
                # Calculate feature-specific similarity (normalized)
                feature_range = self.feature_stats[feature]['max'] - self.feature_stats[feature]['min']
                if feature_range > 0:
                    feature_similarity = 1 - abs(val1 - val2) / feature_range
                    explanations[feature] = feature_similarity
        
        return explanations
    
    def _recommend_by_preferences(
        self,
        user_preferences: Dict[str, float],
        n_recommendations: int
    ) -> List[Tuple[Any, float]]:
        """Recommend whiskeys matching user preferences."""
        recommendations = []
        
        for _, whiskey_row in self.whiskey_features.iterrows():
            whiskey_id = whiskey_row['whiskey_id']
            
            # Calculate preference match score
            match_score = 0
            valid_features = 0
            
            for feature in self.feature_names:
                if feature in user_preferences and pd.notna(whiskey_row[feature]):
                    preferred_value = user_preferences[feature]
                    whiskey_value = whiskey_row[feature]
                    
                    # Calculate match (inverse of normalized difference)
                    feature_range = self.feature_stats[feature]['max'] - self.feature_stats[feature]['min']
                    if feature_range > 0:
                        diff = abs(preferred_value - whiskey_value)
                        match = 1 - (diff / feature_range)
                        match_score += match
                        valid_features += 1
            
            if valid_features > 0:
                avg_match = match_score / valid_features
                recommendations.append((whiskey_id, avg_match))
        
        # Sort by match score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def _recommend_by_feature_quality(self, n_recommendations: int) -> List[Tuple[Any, float]]:
        """Recommend whiskeys with high-quality features."""
        recommendations = []
        
        # Key features that predict high ratings (from linear model analysis)
        key_features = ['complexity', 'finish_duration', 'age']
        
        for _, whiskey_row in self.whiskey_features.iterrows():
            whiskey_id = whiskey_row['whiskey_id']
            
            # Calculate quality score based on key features
            quality_score = 0
            valid_features = 0
            
            for feature in key_features:
                if feature in self.feature_names and pd.notna(whiskey_row[feature]):
                    # Normalize feature to 0-1 scale
                    feature_min = self.feature_stats[feature]['min']
                    feature_max = self.feature_stats[feature]['max']
                    
                    if feature_max > feature_min:
                        normalized_value = (whiskey_row[feature] - feature_min) / (feature_max - feature_min)
                        quality_score += normalized_value
                        valid_features += 1
            
            if valid_features > 0:
                avg_quality = quality_score / valid_features
                recommendations.append((whiskey_id, avg_quality))
        
        # Sort by quality score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def _calculate_feature_bonus(self, whiskey_id: Any) -> float:
        """Calculate rating bonus based on whiskey features."""
        if whiskey_id not in self.whiskey_ids:
            return 0
        
        whiskey_row = self.whiskey_features[self.whiskey_features['whiskey_id'] == whiskey_id].iloc[0]
        
        bonus = 0
        # Complexity bonus (key predictor)
        if 'complexity' in self.feature_names and pd.notna(whiskey_row['complexity']):
            complexity_norm = (whiskey_row['complexity'] - 5) / 5  # Assume 5 is average
            bonus += complexity_norm * 0.5
        
        # Finish duration bonus
        if 'finish_duration' in self.feature_names and pd.notna(whiskey_row['finish_duration']):
            finish_norm = (whiskey_row['finish_duration'] - 5) / 5
            bonus += finish_norm * 0.3
        
        return np.clip(bonus, -2, 2)  # Cap bonus
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Get model-specific data for saving."""
        return {
            'similarity_metric': self.similarity_metric,
            'min_similarity': self.min_similarity,
            'whiskey_features': self.whiskey_features,
            'similarity_matrix': self.similarity_matrix,
            'whiskey_ids': self.whiskey_ids,
            'scaler': self.scaler,
        }
    
    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """Load model-specific data from saved state."""
        self.similarity_metric = data['similarity_metric']
        self.min_similarity = data['min_similarity']
        self.whiskey_features = data['whiskey_features']
        self.similarity_matrix = data['similarity_matrix']
        self.whiskey_ids = data['whiskey_ids']
        self.scaler = data['scaler']


# Convenience function
def train_content_based_recommender(data: pd.DataFrame, **kwargs) -> ContentBasedRecommender:
    """
    Convenience function to train a content-based recommender.
    
    Args:
        data: Training dataset
        **kwargs: Additional arguments for ContentBasedRecommender
        
    Returns:
        Trained ContentBasedRecommender
    """
    model = ContentBasedRecommender(**kwargs)
    return model.fit(data)