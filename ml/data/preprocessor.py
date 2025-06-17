"""
Feature engineering and preprocessing module for WhiskeyHub ML pipeline.

This module handles feature extraction, missing value imputation, scaling,
and creation of user preference profiles for the recommendation system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression

# Set up logging
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocessor handles feature engineering for WhiskeyHub recommendation system.
    
    Key functionality:
    - Missing value imputation (critical: only 52/191 whiskeys have complete features)
    - User preference profile extraction from rating history
    - Feature scaling and normalization
    - Interaction feature creation (user preferences × whiskey attributes)
    - Feature selection based on proven linear model insights
    """
    
    def __init__(
        self,
        whiskey_features: List[str] = None,
        imputation_strategy: str = 'median',
        scaler_type: str = 'standard',
        random_state: int = 42
    ):
        """
        Initialize Preprocessor with configuration.
        
        Args:
            whiskey_features: List of whiskey feature columns to use
            imputation_strategy: Strategy for missing values ('median', 'mean', 'knn')
            scaler_type: Type of feature scaling ('standard', 'minmax', 'none')
            random_state: Random state for reproducible results
        """
        # Core whiskey features based on existing linear model insights
        self.whiskey_features = whiskey_features or [
            'proof', 'price', 'age', 'complexity', 'profiness', 'viscocity', 'finish_duration'
        ]
        
        # Key features that drive ratings (from linear model analysis)
        self.key_features = ['complexity', 'finish_duration', 'age']
        
        self.imputation_strategy = imputation_strategy
        self.scaler_type = scaler_type
        self.random_state = random_state
        
        # Fitted preprocessing components
        self.imputer = None
        self.scaler = None
        self.feature_selector = None
        
        # User preference profiles
        self.user_profiles: Dict[Any, Dict[str, float]] = {}
        
        # Fitted flag
        self.is_fitted = False
        
        logger.info(f"Preprocessor initialized with {len(self.whiskey_features)} whiskey features")
    
    def extract_user_preferences(self, data: pd.DataFrame) -> Dict[Any, Dict[str, float]]:
        """
        Extract user preference profiles from rating history.
        
        For each user, calculate their preferences based on the whiskey features
        they rated highly vs. poorly.
        
        Args:
            data: Merged dataset with user ratings and whiskey features
            
        Returns:
            Dictionary mapping user_id to preference profile
        """
        logger.info("Extracting user preference profiles...")
        
        # Identify user column (could be different names)
        user_col = None
        for col in ['user_id', 'flight_id_pour', 'flight_id']:
            if col in data.columns:
                user_col = col
                break
        
        if user_col is None:
            raise ValueError("No user identifier column found in data")
        
        user_profiles = {}
        
        # Get users with sufficient rating history
        user_counts = data[user_col].value_counts()
        active_users = user_counts[user_counts >= 3].index  # Users with 3+ ratings
        
        for user_id in active_users:
            user_data = data[data[user_col] == user_id].copy()
            
            if len(user_data) < 3:
                continue
            
            # Calculate user's average preference for each feature
            profile = {}
            
            # For each whiskey feature, calculate weighted average based on ratings
            for feature in self.whiskey_features:
                if feature in user_data.columns:
                    # Remove missing values for this calculation
                    valid_data = user_data.dropna(subset=[feature, 'rating'])
                    
                    if len(valid_data) > 0:
                        # Weight by rating (higher rated whiskeys influence preferences more)
                        weights = valid_data['rating'] / valid_data['rating'].max()
                        weighted_avg = np.average(valid_data[feature], weights=weights)
                        profile[feature] = weighted_avg
                    else:
                        profile[feature] = np.nan
            
            # Calculate overall preference strength (how much user varies ratings)
            profile['rating_variance'] = user_data['rating'].var()
            profile['avg_rating'] = user_data['rating'].mean()
            profile['num_ratings'] = len(user_data)
            
            user_profiles[user_id] = profile
        
        self.user_profiles = user_profiles
        logger.info(f"Extracted preferences for {len(user_profiles)} users")
        
        return user_profiles
    
    def impute_missing_values(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values in whiskey features.
        
        Critical for WhiskeyHub: only 52/191 whiskeys have complete feature data.
        
        Args:
            data: DataFrame with potential missing values
            fit: Whether to fit the imputer (True for training, False for prediction)
            
        Returns:
            DataFrame with imputed values
        """
        logger.info(f"Imputing missing values using {self.imputation_strategy} strategy...")
        
        # Get available features in the data
        available_features = [f for f in self.whiskey_features if f in data.columns]
        
        if not available_features:
            logger.warning("No whiskey features found for imputation")
            return data
        
        data_copy = data.copy()
        
        # Initialize imputer if needed
        if fit or self.imputer is None:
            if self.imputation_strategy == 'knn':
                self.imputer = KNNImputer(n_neighbors=5, weights='distance')
            else:
                self.imputer = SimpleImputer(strategy=self.imputation_strategy)
            
            # Fit imputer
            feature_data = data_copy[available_features].select_dtypes(include=[np.number])
            self.imputer.fit(feature_data)
            logger.info(f"Fitted {self.imputation_strategy} imputer on {len(feature_data.columns)} features")
        
        # Apply imputation
        feature_data = data_copy[available_features].select_dtypes(include=[np.number])
        imputed_data = self.imputer.transform(feature_data)
        
        # Update original data
        data_copy[feature_data.columns] = imputed_data
        
        return data_copy
    
    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale whiskey features for ML model compatibility.
        
        Args:
            data: DataFrame with features to scale
            fit: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            DataFrame with scaled features
        """
        if self.scaler_type == 'none':
            return data
        
        logger.info(f"Scaling features using {self.scaler_type} scaler...")
        
        # Get available numeric features
        available_features = [f for f in self.whiskey_features if f in data.columns]
        numeric_features = data[available_features].select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_features:
            logger.warning("No numeric features found for scaling")
            return data
        
        data_copy = data.copy()
        
        # Initialize scaler if needed
        if fit or self.scaler is None:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            
            # Fit scaler
            self.scaler.fit(data_copy[numeric_features])
            logger.info(f"Fitted {self.scaler_type} scaler on {len(numeric_features)} features")
        
        # Apply scaling
        scaled_data = self.scaler.transform(data_copy[numeric_features])
        data_copy[numeric_features] = scaled_data
        
        return data_copy
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between user preferences and whiskey attributes.
        
        These features capture how well a whiskey matches each user's preferences.
        
        Args:
            data: DataFrame with user and whiskey data
            
        Returns:
            DataFrame with additional interaction features
        """
        logger.info("Creating user-whiskey interaction features...")
        
        if not self.user_profiles:
            logger.warning("No user profiles available. Run extract_user_preferences first.")
            return data
        
        # Identify user column
        user_col = None
        for col in ['user_id', 'flight_id_pour', 'flight_id']:
            if col in data.columns:
                user_col = col
                break
        
        if user_col is None:
            logger.warning("No user identifier found for interaction features")
            return data
        
        data_copy = data.copy()
        
        # Create interaction features for key features
        for feature in self.key_features:
            if feature in data.columns:
                interaction_col = f'user_{feature}_match'
                
                # Calculate match score for each row
                match_scores = []
                for _, row in data_copy.iterrows():
                    user_id = row[user_col]
                    whiskey_value = row[feature]
                    
                    if user_id in self.user_profiles and not pd.isna(whiskey_value):
                        user_pref = self.user_profiles[user_id].get(feature, np.nan)
                        if not pd.isna(user_pref):
                            # Calculate match as inverse of absolute difference (normalized)
                            diff = abs(whiskey_value - user_pref)
                            max_diff = data_copy[feature].max() - data_copy[feature].min()
                            match_score = 1 - (diff / max_diff) if max_diff > 0 else 1
                        else:
                            match_score = 0.5  # Neutral match
                    else:
                        match_score = 0.5  # Neutral match for unknown users/values
                    
                    match_scores.append(match_score)
                
                data_copy[interaction_col] = match_scores
        
        logger.info(f"Created {len(self.key_features)} interaction features")
        return data_copy
    
    def select_features(self, data: pd.DataFrame, target_col: str = 'rating', k: int = 10) -> pd.DataFrame:
        """
        Select top k features based on statistical tests.
        
        Uses insights from existing linear model analysis.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting top {k} features...")
        
        # Get feature columns (exclude target and identifiers)
        exclude_cols = [target_col, 'user_id', 'flight_id_pour', 'flight_id', 'whiskey_id', 'id', 'id_pour', 'id_note']
        feature_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) <= k:
            logger.info(f"Only {len(feature_cols)} features available, using all")
            return data
        
        # Prepare data for feature selection
        X = data[feature_cols].fillna(0)  # Simple fillna for feature selection
        y = data[target_col]
        
        # Initialize feature selector
        self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        
        # Fit and select features
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_feature_names = np.array(feature_cols)[self.feature_selector.get_support()]
        
        logger.info(f"Selected features: {list(selected_feature_names)}")
        
        # Return data with selected features plus essential columns
        essential_cols = [col for col in exclude_cols if col in data.columns]
        return data[essential_cols + list(selected_feature_names)]
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform training data.
        
        Args:
            data: Training dataset
            
        Returns:
            Preprocessed training data
        """
        logger.info("Fitting and transforming training data...")
        
        # Step 1: Extract user preferences
        self.extract_user_preferences(data)
        
        # Step 2: Handle missing values
        data = self.impute_missing_values(data, fit=True)
        
        # Step 3: Create interaction features
        data = self.create_interaction_features(data)
        
        # Step 4: Scale features
        data = self.scale_features(data, fit=True)
        
        self.is_fitted = True
        logger.info("Preprocessing pipeline fitted successfully")
        
        return data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            data: New dataset to transform
            
        Returns:
            Preprocessed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        logger.info("Transforming new data...")
        
        # Apply same preprocessing steps (without fitting)
        data = self.impute_missing_values(data, fit=False)
        data = self.create_interaction_features(data)
        data = self.scale_features(data, fit=False)
        
        return data
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about processed features.
        
        Returns:
            Dictionary with feature information
        """
        info = {
            'whiskey_features': self.whiskey_features,
            'key_features': self.key_features,
            'num_user_profiles': len(self.user_profiles),
            'is_fitted': self.is_fitted,
            'imputation_strategy': self.imputation_strategy,
            'scaler_type': self.scaler_type,
        }
        
        if self.is_fitted:
            info['has_imputer'] = self.imputer is not None
            info['has_scaler'] = self.scaler is not None
            info['has_feature_selector'] = self.feature_selector is not None
        
        return info


# Convenience function for quick preprocessing
def preprocess_whiskeyhub_data(
    data: pd.DataFrame,
    whiskey_features: List[str] = None,
    test_data: pd.DataFrame = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Convenience function to preprocess WhiskeyHub data.
    
    Args:
        data: Training dataset
        whiskey_features: List of whiskey features to use
        test_data: Optional test dataset
        
    Returns:
        Tuple of (processed_train_data, processed_test_data)
    """
    preprocessor = Preprocessor(whiskey_features=whiskey_features)
    
    # Process training data
    train_processed = preprocessor.fit_transform(data)
    
    # Process test data if provided
    test_processed = None
    if test_data is not None:
        test_processed = preprocessor.transform(test_data)
    
    return train_processed, test_processed