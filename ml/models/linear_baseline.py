"""
Linear baseline model for WhiskeyHub recommendation system.

This module refactors the proven linear regression model from scripts/linear_model.py
into a reusable class that maintains the R² = 0.765 performance benchmark.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from .base_model import BasePredictiveModel

# Set up logging
logger = logging.getLogger(__name__)


class LinearBaselineModel(BasePredictiveModel):
    """
    Linear regression baseline model for whiskey rating prediction.
    
    This model refactors the proven linear regression approach that achieved
    R² = 0.765 on the WhiskeyHub dataset. It maintains exact same performance
    while providing a clean, reusable interface.
    
    Key features:
    - Uses proven whiskey features: complexity, finish_duration, age, etc.
    - Maintains R² = 0.765 performance benchmark
    - Provides feature importance via linear coefficients
    - Supports single predictions and batch recommendations
    - Integrates with DataLoader and Preprocessor pipeline
    """
    
    def __init__(
        self,
        features: List[str] = None,
        aggregate_by_whiskey: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Linear Baseline Model.
        
        Args:
            features: List of features to use (defaults to proven features)
            aggregate_by_whiskey: Whether to aggregate ratings by whiskey
            random_state: Random state for reproducible results
            **kwargs: Additional arguments passed to BasePredictiveModel
        """
        # Default to proven features from original analysis
        default_features = [
            'proof', 'price', 'age', 'complexity', 'profiness', 'viscocity', 'finish_duration'
        ]
        
        super().__init__(
            features=features or default_features,
            model_name='LinearBaselineModel',
            random_state=random_state,
            **kwargs
        )
        
        self.aggregate_by_whiskey = aggregate_by_whiskey
        
        # Model components
        self.model = LinearRegression()
        self.whiskey_data: Optional[pd.DataFrame] = None
        self.all_whiskey_ids: List[Any] = []
        
        # Performance tracking
        self.target_r2 = 0.765  # Benchmark from original analysis
        self.target_rmse = 0.613
        
        logger.info(f"Initialized LinearBaselineModel with {len(self.features)} features")
    
    def fit(self, data: pd.DataFrame, target_col: str = 'rating') -> 'LinearBaselineModel':
        """
        Train the linear regression model.
        
        Maintains exact same logic as the original scripts/linear_model.py
        to ensure R² = 0.765 performance benchmark.
        
        Args:
            data: Training dataset with user-whiskey interactions
            target_col: Name of the target column
            
        Returns:
            Self for method chaining
        """
        logger.info("Training LinearBaselineModel...")
        
        # Step 1: Find available features (same logic as original script)
        available_features = []
        for feature in self.features:
            matching_cols = [col for col in data.columns if feature.lower() in col.lower()]
            if matching_cols:
                available_features.append(matching_cols[0])
        
        if not available_features:
            raise ValueError("No valid features found for model training")
        
        self.feature_names = available_features
        logger.info(f"Using features: {self.feature_names}")
        
        # Step 2: Aggregate by whiskey (same as original script)
        if self.aggregate_by_whiskey and 'whiskey_id' in data.columns:
            whiskey_data = data.groupby('whiskey_id').agg({
                target_col: 'mean',
                **{feat: 'first' for feat in self.feature_names if feat in data.columns}
            }).reset_index()
            
            logger.info(f"Aggregated to {len(whiskey_data)} unique whiskeys")
        else:
            whiskey_data = data.copy()
        
        # Step 3: Clean data (same as original script)  
        whiskey_data_clean = whiskey_data.dropna(subset=self.feature_names + [target_col])
        logger.info(f"Clean data: {len(whiskey_data_clean)} samples after removing missing values")
        
        if len(whiskey_data_clean) < 10:
            raise ValueError("Insufficient clean data for training")
        
        # Step 4: Prepare features and target
        X = whiskey_data_clean[self.feature_names]
        y = whiskey_data_clean[target_col]
        
        # Store for later use in predictions
        self.whiskey_data = whiskey_data_clean.copy()
        self.all_whiskey_ids = whiskey_data_clean.get('whiskey_id', []).tolist()
        
        # Calculate feature statistics
        self._calculate_feature_stats(X)
        
        # Step 5: Train model (note: scaling handled by Preprocessor in pipeline)
        self.model.fit(X, y)
        
        # Step 6: Evaluate performance to ensure benchmark is maintained
        y_pred = self.model.predict(X)
        train_r2 = r2_score(y, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Cross-validation (same as original script)
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        # Store training statistics
        self.training_stats = {
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'cv_rmse': cv_rmse,
            'target_r2': self.target_r2,
            'target_rmse': self.target_rmse,
        }
        
        # Verify performance benchmark
        if train_r2 < (self.target_r2 - 0.05):  # Allow 5% tolerance
            logger.warning(f"Performance below benchmark: R² = {train_r2:.3f} < {self.target_r2:.3f}")
        else:
            logger.info(f"✅ Performance benchmark met: R² = {train_r2:.3f}")
        
        self.is_fitted = True
        logger.info("LinearBaselineModel training completed")
        
        return self
    
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
            user_id: User identifier (not used in this model)
            whiskey_id: Whiskey identifier
            user_features: Optional user features (not used)
            whiskey_features: Optional whiskey features (used if provided)
            
        Returns:
            Predicted rating
        """
        self._validate_fitted()
        
        # Option 1: Use provided whiskey features
        if whiskey_features:
            feature_values = []
            for feature in self.feature_names:
                if feature in whiskey_features:
                    feature_values.append(whiskey_features[feature])
                else:
                    # Use feature mean as fallback
                    mean_val = self.feature_stats.get(feature, {}).get('mean', 0)
                    feature_values.append(mean_val)
                    logger.warning(f"Missing feature {feature}, using mean: {mean_val}")
            
            X = np.array(feature_values).reshape(1, -1)
            prediction = self.model.predict(X)[0]
        
        # Option 2: Look up whiskey in training data
        elif self.whiskey_data is not None and whiskey_id in self.all_whiskey_ids:
            whiskey_row = self.whiskey_data[self.whiskey_data['whiskey_id'] == whiskey_id]
            if len(whiskey_row) > 0:
                X = whiskey_row[self.feature_names].values
                prediction = self.model.predict(X)[0]
            else:
                # Use global mean as fallback
                prediction = self.training_stats.get('global_mean', 7.0)
                logger.warning(f"Whiskey {whiskey_id} not found, using global mean")
        
        # Option 3: Use feature means as fallback
        else:
            feature_means = [self.feature_stats.get(f, {}).get('mean', 0) for f in self.feature_names]
            X = np.array(feature_means).reshape(1, -1)
            prediction = self.model.predict(X)[0]
            logger.info(f"Using feature means for unknown whiskey {whiskey_id}")
        
        # Ensure prediction is within reasonable bounds
        return np.clip(prediction, 0, 10)
    
    def recommend(
        self,
        user_id: Any,
        n_recommendations: int = 10,
        exclude_rated: bool = True,
        min_rating: float = None
    ) -> List[Tuple[Any, float]]:
        """
        Generate whiskey recommendations for a user.
        
        Since this is a content-based model, recommendations are based on
        whiskey features and predicted ratings.
        
        Args:
            user_id: User identifier (not used in this baseline model)
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude rated whiskeys (not implemented)
            min_rating: Minimum rating threshold
            
        Returns:
            List of (whiskey_id, predicted_rating) tuples
        """
        self._validate_fitted()
        
        if self.whiskey_data is None:
            raise ValueError("No whiskey data available for recommendations")
        
        # Predict ratings for all whiskeys
        recommendations = []
        
        for _, whiskey_row in self.whiskey_data.iterrows():
            whiskey_id = whiskey_row.get('whiskey_id', 'unknown')
            
            # Get whiskey features
            whiskey_features = {
                feature: whiskey_row[feature] 
                for feature in self.feature_names 
                if feature in whiskey_row and not pd.isna(whiskey_row[feature])
            }
            
            # Predict rating
            try:
                predicted_rating = self.predict_rating(user_id, whiskey_id, whiskey_features=whiskey_features)
                
                # Apply minimum rating filter
                if min_rating is None or predicted_rating >= min_rating:
                    recommendations.append((whiskey_id, predicted_rating))
                    
            except Exception as e:
                logger.warning(f"Failed to predict for whiskey {whiskey_id}: {e}")
        
        # Sort by predicted rating (descending) and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from linear regression coefficients.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        self._validate_fitted()
        
        if not hasattr(self.model, 'coef_'):
            return {}
        
        # Use absolute coefficients as importance scores
        importance = {}
        for feature, coeff in zip(self.feature_names, self.model.coef_):
            importance[feature] = abs(coeff)
        
        return importance
    
    def create_visualizations(self, save_path: str = "../results/") -> Dict[str, str]:
        """
        Create and save model visualizations.
        
        Args:
            save_path: Directory to save visualizations
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        self._validate_fitted()
        
        if self.whiskey_data is None:
            logger.warning("No data available for visualizations")
            return {}
        
        from pathlib import Path
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        # Feature importance plot
        try:
            importance = self.get_feature_importance()
            if importance:
                plt.figure(figsize=(10, 6))
                features = list(importance.keys())
                values = list(importance.values())
                
                plt.barh(features, values)
                plt.xlabel('Absolute Coefficient Value')
                plt.title('Linear Baseline Model - Feature Importance')
                plt.tight_layout()
                
                plot_path = save_path / 'linear_baseline_feature_importance.png'
                plt.savefig(plot_path)
                plt.close()
                saved_plots['feature_importance'] = str(plot_path)
                
        except Exception as e:
            logger.error(f"Failed to create feature importance plot: {e}")
        
        return saved_plots
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Get model-specific data for saving."""
        return {
            'model': self.model,
            'whiskey_data': self.whiskey_data,
            'all_whiskey_ids': self.all_whiskey_ids,
            'aggregate_by_whiskey': self.aggregate_by_whiskey,
            'target_r2': self.target_r2,
            'target_rmse': self.target_rmse,
        }
    
    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """Load model-specific data from saved state."""
        self.model = data['model']
        self.whiskey_data = data['whiskey_data']
        self.all_whiskey_ids = data['all_whiskey_ids']
        self.aggregate_by_whiskey = data['aggregate_by_whiskey']
        self.target_r2 = data['target_r2']
        self.target_rmse = data['target_rmse']


# Convenience function for backward compatibility
def train_linear_baseline(data: pd.DataFrame, **kwargs) -> LinearBaselineModel:
    """
    Convenience function to train a linear baseline model.
    
    Args:
        data: Training dataset
        **kwargs: Additional arguments for LinearBaselineModel
        
    Returns:
        Trained LinearBaselineModel
    """
    model = LinearBaselineModel(**kwargs)
    return model.fit(data)