"""
Base model classes for WhiskeyHub recommendation system.

This module defines abstract base classes that provide consistent interfaces
for all recommendation models in the WhiskeyHub ML pipeline.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import pickle
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """
    Abstract base class for all WhiskeyHub recommendation models.
    
    This class defines the standard interface that all recommendation models
    must implement, ensuring consistency across different approaches
    (content-based, collaborative, hybrid).
    """
    
    def __init__(self, model_name: str = None, random_state: int = 42):
        """
        Initialize base recommender.
        
        Args:
            model_name: Name identifier for the model
            random_state: Random state for reproducible results
        """
        self.model_name = model_name or self.__class__.__name__
        self.random_state = random_state
        self.is_fitted = False
        
        # Model metadata
        self.feature_names: List[str] = []
        self.training_stats: Dict[str, Any] = {}
        
        logger.info(f"Initialized {self.model_name}")
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target_col: str = 'rating') -> 'BaseRecommender':
        """
        Train the recommendation model.
        
        Args:
            data: Training dataset with user-whiskey interactions
            target_col: Name of the target column (ratings)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict_rating(
        self, 
        user_id: Any, 
        whiskey_id: Any, 
        user_features: Dict[str, Any] = None,
        whiskey_features: Dict[str, Any] = None
    ) -> float:
        """
        Predict rating for a specific user-whiskey pair.
        
        Args:
            user_id: User identifier
            whiskey_id: Whiskey identifier  
            user_features: Optional user feature dictionary
            whiskey_features: Optional whiskey feature dictionary
            
        Returns:
            Predicted rating (typically 0-10 scale)
        """
        pass
    
    @abstractmethod
    def recommend(
        self, 
        user_id: Any, 
        n_recommendations: int = 10,
        exclude_rated: bool = True,
        min_rating: float = None
    ) -> List[Tuple[Any, float]]:
        """
        Generate personalized whiskey recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude already-rated whiskeys
            min_rating: Minimum predicted rating threshold
            
        Returns:
            List of (whiskey_id, predicted_rating) tuples, sorted by rating
        """
        pass
    
    def predict_batch(
        self, 
        user_whiskey_pairs: List[Tuple[Any, Any]]
    ) -> List[float]:
        """
        Predict ratings for multiple user-whiskey pairs.
        
        Args:
            user_whiskey_pairs: List of (user_id, whiskey_id) tuples
            
        Returns:
            List of predicted ratings
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} not fitted. Call fit() first.")
        
        predictions = []
        for user_id, whiskey_id in user_whiskey_pairs:
            try:
                prediction = self.predict_rating(user_id, whiskey_id)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Prediction failed for user {user_id}, whiskey {whiskey_id}: {e}")
                predictions.append(np.nan)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Default implementation returns empty dict
        # Subclasses should override this if they support feature importance
        logger.warning(f"{self.model_name} does not support feature importance")
        return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary with model metadata and statistics
        """
        info = {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
        }
        return info
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} not fitted. Call fit() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_state = {
            'model_name': self.model_name,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'model_data': self._get_save_data()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        
        logger.info(f"Saved {self.model_name} to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> 'BaseRecommender':
        """
        Load a fitted model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        # Restore model state
        self.model_name = model_state['model_name']
        self.random_state = model_state['random_state']
        self.feature_names = model_state['feature_names']
        self.training_stats = model_state['training_stats']
        
        # Load model-specific data
        self._load_save_data(model_state['model_data'])
        self.is_fitted = True
        
        logger.info(f"Loaded {self.model_name} from {filepath}")
        return self
    
    def _get_save_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.
        
        Subclasses should override this to save their specific model data.
        
        Returns:
            Dictionary with model-specific data
        """
        return {}
    
    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """
        Load model-specific data from saved state.
        
        Subclasses should override this to restore their specific model data.
        
        Args:
            data: Dictionary with model-specific data
        """
        pass
    
    def _validate_fitted(self) -> None:
        """Check if model is fitted, raise error if not."""
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} not fitted. Call fit() first.")


class BasePredictiveModel(BaseRecommender):
    """
    Base class for predictive models that use features to predict ratings.
    
    This extends BaseRecommender with functionality specific to models
    that use whiskey/user features for prediction (like linear regression,
    content-based filtering).
    """
    
    def __init__(self, features: List[str] = None, **kwargs):
        """
        Initialize predictive model.
        
        Args:
            features: List of feature names to use for prediction
            **kwargs: Additional arguments passed to BaseRecommender
        """
        super().__init__(**kwargs)
        self.features = features or []
        self.feature_stats: Dict[str, Dict[str, float]] = {}
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix for model training/prediction.
        
        Args:
            data: Dataset with features
            
        Returns:
            Feature matrix with selected features
        """
        # Use specified features or detect numeric features
        if self.features:
            available_features = [f for f in self.features if f in data.columns]
        else:
            available_features = data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target and identifier columns
            exclude_cols = ['rating', 'user_id', 'whiskey_id', 'id', 'flight_id']
            available_features = [f for f in available_features if f not in exclude_cols]
        
        if not available_features:
            raise ValueError("No valid features found for model training")
        
        self.feature_names = available_features
        return data[available_features]
    
    def _calculate_feature_stats(self, data: pd.DataFrame) -> None:
        """Calculate and store feature statistics."""
        for feature in self.feature_names:
            if feature in data.columns:
                self.feature_stats[feature] = {
                    'mean': data[feature].mean(),
                    'std': data[feature].std(),
                    'min': data[feature].min(),
                    'max': data[feature].max(),
                    'missing_pct': data[feature].isnull().mean()
                }


class BaseCollaborativeModel(BaseRecommender):
    """
    Base class for collaborative filtering models.
    
    This extends BaseRecommender with functionality specific to collaborative
    filtering approaches that use user-item interaction patterns.
    """
    
    def __init__(self, **kwargs):
        """Initialize collaborative filtering model."""
        super().__init__(**kwargs)
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_similarities: Optional[pd.DataFrame] = None
        self.item_similarities: Optional[pd.DataFrame] = None
    
    def _create_user_item_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-item interaction matrix.
        
        Args:
            data: Dataset with user-item interactions
            
        Returns:
            User-item matrix with ratings
        """
        # Identify user and item columns
        user_col = self._get_user_column(data)
        item_col = self._get_item_column(data)
        
        # Create pivot table
        matrix = data.pivot_table(
            index=user_col,
            columns=item_col,
            values='rating',
            aggfunc='mean'  # Handle multiple ratings per user-item pair
        )
        
        self.user_item_matrix = matrix
        logger.info(f"Created user-item matrix: {matrix.shape}")
        return matrix
    
    def _get_user_column(self, data: pd.DataFrame) -> str:
        """Identify user column in dataset."""
        user_cols = ['user_id', 'flight_id_pour', 'flight_id']
        for col in user_cols:
            if col in data.columns:
                return col
        raise ValueError("No user identifier column found")
    
    def _get_item_column(self, data: pd.DataFrame) -> str:
        """Identify item (whiskey) column in dataset."""
        item_cols = ['whiskey_id', 'item_id']
        for col in item_cols:
            if col in data.columns:
                return col
        raise ValueError("No item identifier column found")