"""
Hybrid recommendation model for WhiskeyHub.

This model combines collaborative filtering and content-based filtering
to provide the best possible recommendations by leveraging both user behavior
patterns and whiskey feature similarities.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Union

from .base_model import BaseRecommender
from .collaborative import CollaborativeRecommender
from .content_based import ContentBasedRecommender
from .linear_baseline import LinearBaselineModel

# Set up logging
logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommendation model combining multiple approaches.
    
    This model intelligently combines:
    - Collaborative filtering (leverages 58% data density)
    - Content-based filtering (uses proven whiskey features)
    - Linear baseline (R² = 0.775 performance)
    
    Key features:
    - Dynamic weight adjustment based on data availability
    - Fallback strategies for cold start problems
    - Ensemble methods for improved accuracy
    - Configurable weighting schemes
    """
    
    def __init__(
        self,
        collaborative_weight: float = 0.6,
        content_weight: float = 0.3,
        baseline_weight: float = 0.1,
        min_collaborative_ratings: int = 3,
        min_content_features: int = 3,
        adaptive_weighting: bool = True,
        **kwargs
    ):
        """
        Initialize Hybrid Recommender.
        
        Args:
            collaborative_weight: Weight for collaborative filtering
            content_weight: Weight for content-based filtering
            baseline_weight: Weight for linear baseline
            min_collaborative_ratings: Min ratings needed for collaborative
            min_content_features: Min features needed for content-based
            adaptive_weighting: Whether to adjust weights dynamically
            **kwargs: Additional arguments passed to BaseRecommender
        """
        super().__init__(
            model_name='HybridRecommender',
            **kwargs
        )
        
        # Validate weights sum to 1.0
        total_weight = collaborative_weight + content_weight + baseline_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            collaborative_weight /= total_weight
            content_weight /= total_weight
            baseline_weight /= total_weight
        
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.baseline_weight = baseline_weight
        self.min_collaborative_ratings = min_collaborative_ratings
        self.min_content_features = min_content_features
        self.adaptive_weighting = adaptive_weighting
        
        # Component models
        self.collaborative_model: Optional[CollaborativeRecommender] = None
        self.content_model: Optional[ContentBasedRecommender] = None
        self.baseline_model: Optional[LinearBaselineModel] = None
        
        # Model availability tracking
        self.models_available = {
            'collaborative': False,
            'content': False,
            'baseline': False
        }
        
        logger.info(f"Initialized HybridRecommender with weights: "
                   f"collab={collaborative_weight:.2f}, content={content_weight:.2f}, "
                   f"baseline={baseline_weight:.2f}")
    
    def fit(self, data: pd.DataFrame, target_col: str = 'rating') -> 'HybridRecommender':
        """
        Train all component models for the hybrid system.
        
        Args:
            data: Training dataset
            target_col: Name of the target column
            
        Returns:
            Self for method chaining
        """
        logger.info("Training HybridRecommender components...")
        
        # Train collaborative filtering model
        try:
            logger.info("Training collaborative filtering component...")
            self.collaborative_model = CollaborativeRecommender(
                approach='user_based',
                similarity_metric='cosine',
                n_neighbors=20,
                min_similarity=0.1
            )
            self.collaborative_model.fit(data, target_col)
            self.models_available['collaborative'] = True
            logger.info("✅ Collaborative model trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Collaborative model training failed: {e}")
            self.models_available['collaborative'] = False
        
        # Train content-based model
        try:
            logger.info("Training content-based component...")
            self.content_model = ContentBasedRecommender(
                similarity_metric='cosine',
                min_similarity=0.1
            )
            self.content_model.fit(data, target_col)
            self.models_available['content'] = True
            logger.info("✅ Content-based model trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Content-based model training failed: {e}")
            self.models_available['content'] = False
        
        # Train linear baseline model
        try:
            logger.info("Training linear baseline component...")
            self.baseline_model = LinearBaselineModel()
            self.baseline_model.fit(data, target_col)
            self.models_available['baseline'] = True
            logger.info("✅ Linear baseline model trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Linear baseline training failed: {e}")
            self.models_available['baseline'] = False
        
        # Check if at least one model is available
        if not any(self.models_available.values()):
            raise ValueError("No component models could be trained successfully")
        
        # Store training statistics
        self.training_stats = {
            'models_available': self.models_available,
            'collaborative_weight': self.collaborative_weight,
            'content_weight': self.content_weight,
            'baseline_weight': self.baseline_weight,
            'adaptive_weighting': self.adaptive_weighting,
        }
        
        # Add component model stats if available
        if self.models_available['collaborative']:
            self.training_stats['collaborative_stats'] = self.collaborative_model.training_stats
        if self.models_available['content']:
            self.training_stats['content_stats'] = self.content_model.training_stats
        if self.models_available['baseline']:
            self.training_stats['baseline_stats'] = self.baseline_model.training_stats
        
        self.is_fitted = True
        logger.info("HybridRecommender training completed")
        
        return self
    
    def predict_rating(
        self,
        user_id: Any,
        whiskey_id: Any,
        user_features: Dict[str, Any] = None,
        whiskey_features: Dict[str, Any] = None
    ) -> float:
        """
        Predict rating using hybrid approach.
        
        Args:
            user_id: User identifier
            whiskey_id: Whiskey identifier
            user_features: Optional user features
            whiskey_features: Optional whiskey features
            
        Returns:
            Predicted rating
        """
        self._validate_fitted()
        
        predictions = {}
        weights = {}
        
        # Get predictions from available models
        if self.models_available['collaborative']:
            try:
                collab_pred = self.collaborative_model.predict_rating(
                    user_id, whiskey_id, user_features, whiskey_features
                )
                predictions['collaborative'] = collab_pred
                weights['collaborative'] = self.collaborative_weight
            except Exception as e:
                logger.debug(f"Collaborative prediction failed: {e}")
        
        if self.models_available['content']:
            try:
                content_pred = self.content_model.predict_rating(
                    user_id, whiskey_id, user_features, whiskey_features
                )
                predictions['content'] = content_pred
                weights['content'] = self.content_weight
            except Exception as e:
                logger.debug(f"Content-based prediction failed: {e}")
        
        if self.models_available['baseline']:
            try:
                baseline_pred = self.baseline_model.predict_rating(
                    user_id, whiskey_id, user_features, whiskey_features
                )
                predictions['baseline'] = baseline_pred
                weights['baseline'] = self.baseline_weight
            except Exception as e:
                logger.debug(f"Baseline prediction failed: {e}")
        
        if not predictions:
            logger.warning("No models could make predictions, using global fallback")
            return 7.0  # Global average fallback
        
        # Adjust weights dynamically if enabled
        if self.adaptive_weighting:
            weights = self._adjust_weights_dynamically(
                user_id, whiskey_id, weights, user_features, whiskey_features
            )
        
        # Calculate weighted average
        weighted_sum = sum(pred * weights.get(model, 0) for model, pred in predictions.items())
        total_weight = sum(weights.get(model, 0) for model in predictions.keys())
        
        if total_weight > 0:
            final_prediction = weighted_sum / total_weight
        else:
            # Equal weighting fallback
            final_prediction = sum(predictions.values()) / len(predictions)
        
        return np.clip(final_prediction, 0, 10)
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_rated: bool = True,
        min_rating: float = None
    ) -> List[Tuple[Any, float]]:
        """
        Generate hybrid recommendations.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude already-rated items
            min_rating: Minimum predicted rating threshold
            
        Returns:
            List of (whiskey_id, predicted_rating) tuples
        """
        self._validate_fitted()
        
        # Get recommendations from each available model
        all_recommendations = {}
        
        if self.models_available['collaborative']:
            try:
                collab_recs = self.collaborative_model.recommend(
                    user_id, n_recommendations * 2, exclude_rated, min_rating
                )
                for whiskey_id, score in collab_recs:
                    if whiskey_id not in all_recommendations:
                        all_recommendations[whiskey_id] = {}
                    all_recommendations[whiskey_id]['collaborative'] = score
            except Exception as e:
                logger.debug(f"Collaborative recommendations failed: {e}")
        
        if self.models_available['content']:
            try:
                content_recs = self.content_model.recommend(
                    user_id, n_recommendations * 2, exclude_rated, min_rating
                )
                for whiskey_id, score in content_recs:
                    if whiskey_id not in all_recommendations:
                        all_recommendations[whiskey_id] = {}
                    all_recommendations[whiskey_id]['content'] = score
            except Exception as e:
                logger.debug(f"Content-based recommendations failed: {e}")
        
        if self.models_available['baseline']:
            try:
                baseline_recs = self.baseline_model.recommend(
                    user_id, n_recommendations * 2, exclude_rated, min_rating
                )
                for whiskey_id, score in baseline_recs:
                    if whiskey_id not in all_recommendations:
                        all_recommendations[whiskey_id] = {}
                    all_recommendations[whiskey_id]['baseline'] = score
            except Exception as e:
                logger.debug(f"Baseline recommendations failed: {e}")
        
        # Combine recommendations using hybrid scoring
        final_recommendations = []
        
        for whiskey_id, model_scores in all_recommendations.items():
            # Calculate hybrid score
            hybrid_score = 0
            total_weight = 0
            
            for model, score in model_scores.items():
                if model == 'collaborative' and self.models_available['collaborative']:
                    weight = self.collaborative_weight
                elif model == 'content' and self.models_available['content']:
                    weight = self.content_weight
                elif model == 'baseline' and self.models_available['baseline']:
                    weight = self.baseline_weight
                else:
                    continue
                
                hybrid_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                final_score = hybrid_score / total_weight
                
                # Apply minimum rating filter
                if min_rating is None or final_score >= min_rating:
                    final_recommendations.append((whiskey_id, final_score))
        
        # Sort by hybrid score and return top N
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        return final_recommendations[:n_recommendations]
    
    def _adjust_weights_dynamically(
        self,
        user_id: Any,
        whiskey_id: Any,
        base_weights: Dict[str, float],
        user_features: Dict[str, Any] = None,
        whiskey_features: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Dynamically adjust model weights based on data availability and quality.
        
        Args:
            user_id: User identifier
            whiskey_id: Whiskey identifier
            base_weights: Base weights for each model
            user_features: Optional user features
            whiskey_features: Optional whiskey features
            
        Returns:
            Adjusted weights dictionary
        """
        adjusted_weights = base_weights.copy()
        
        # Check collaborative filtering data availability
        if 'collaborative' in adjusted_weights and self.models_available['collaborative']:
            user_rating_count = 0
            if hasattr(self.collaborative_model, 'user_item_matrix'):
                if user_id in self.collaborative_model.user_item_matrix.index:
                    user_ratings = self.collaborative_model.user_item_matrix.loc[user_id]
                    user_rating_count = user_ratings.notna().sum()
            
            # Reduce collaborative weight for users with few ratings
            if user_rating_count < self.min_collaborative_ratings:
                adjusted_weights['collaborative'] *= 0.5
                logger.debug(f"Reduced collaborative weight for user {user_id} "
                           f"({user_rating_count} ratings)")
        
        # Check content-based data availability
        if 'content' in adjusted_weights and self.models_available['content']:
            whiskey_feature_count = 0
            if hasattr(self.content_model, 'whiskey_features'):
                if self.content_model.whiskey_features is not None:
                    whiskey_row = self.content_model.whiskey_features[
                        self.content_model.whiskey_features['whiskey_id'] == whiskey_id
                    ]
                    if len(whiskey_row) > 0:
                        whiskey_feature_count = whiskey_row.iloc[0].notna().sum()
            
            # Reduce content weight for whiskeys with few features
            if whiskey_feature_count < self.min_content_features:
                adjusted_weights['content'] *= 0.5
                logger.debug(f"Reduced content weight for whiskey {whiskey_id} "
                           f"({whiskey_feature_count} features)")
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for model in adjusted_weights:
                adjusted_weights[model] /= total_weight
        
        return adjusted_weights
    
    def get_model_contributions(
        self,
        user_id: Any,
        whiskey_id: Any
    ) -> Dict[str, Dict[str, float]]:
        """
        Get detailed breakdown of each model's contribution to the prediction.
        
        Args:
            user_id: User identifier
            whiskey_id: Whiskey identifier
            
        Returns:
            Dictionary with model contributions and weights
        """
        self._validate_fitted()
        
        contributions = {}
        
        # Get individual predictions and weights
        if self.models_available['collaborative']:
            try:
                pred = self.collaborative_model.predict_rating(user_id, whiskey_id)
                contributions['collaborative'] = {
                    'prediction': pred,
                    'weight': self.collaborative_weight,
                    'contribution': pred * self.collaborative_weight
                }
            except Exception as e:
                logger.debug(f"Could not get collaborative contribution: {e}")
        
        if self.models_available['content']:
            try:
                pred = self.content_model.predict_rating(user_id, whiskey_id)
                contributions['content'] = {
                    'prediction': pred,
                    'weight': self.content_weight,
                    'contribution': pred * self.content_weight
                }
            except Exception as e:
                logger.debug(f"Could not get content contribution: {e}")
        
        if self.models_available['baseline']:
            try:
                pred = self.baseline_model.predict_rating(user_id, whiskey_id)
                contributions['baseline'] = {
                    'prediction': pred,
                    'weight': self.baseline_weight,
                    'contribution': pred * self.baseline_weight
                }
            except Exception as e:
                logger.debug(f"Could not get baseline contribution: {e}")
        
        return contributions
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Get model-specific data for saving."""
        return {
            'collaborative_weight': self.collaborative_weight,
            'content_weight': self.content_weight,
            'baseline_weight': self.baseline_weight,
            'min_collaborative_ratings': self.min_collaborative_ratings,
            'min_content_features': self.min_content_features,
            'adaptive_weighting': self.adaptive_weighting,
            'models_available': self.models_available,
            'collaborative_model': self.collaborative_model,
            'content_model': self.content_model,
            'baseline_model': self.baseline_model,
        }
    
    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """Load model-specific data from saved state."""
        self.collaborative_weight = data['collaborative_weight']
        self.content_weight = data['content_weight']
        self.baseline_weight = data['baseline_weight']
        self.min_collaborative_ratings = data['min_collaborative_ratings']
        self.min_content_features = data['min_content_features']
        self.adaptive_weighting = data['adaptive_weighting']
        self.models_available = data['models_available']
        self.collaborative_model = data['collaborative_model']
        self.content_model = data['content_model']
        self.baseline_model = data['baseline_model']


# Convenience function
def train_hybrid_recommender(
    data: pd.DataFrame,
    collaborative_weight: float = 0.6,
    content_weight: float = 0.3,
    baseline_weight: float = 0.1,
    **kwargs
) -> HybridRecommender:
    """
    Convenience function to train a hybrid recommender.
    
    Args:
        data: Training dataset
        collaborative_weight: Weight for collaborative filtering
        content_weight: Weight for content-based filtering
        baseline_weight: Weight for linear baseline
        **kwargs: Additional arguments for HybridRecommender
        
    Returns:
        Trained HybridRecommender
    """
    model = HybridRecommender(
        collaborative_weight=collaborative_weight,
        content_weight=content_weight,
        baseline_weight=baseline_weight,
        **kwargs
    )
    return model.fit(data)