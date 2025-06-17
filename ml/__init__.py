"""
WhiskeyHub Machine Learning Pipeline

A hybrid recommendation system combining collaborative filtering and content-based 
approaches for whiskey recommendations, leveraging exceptional 58% data density.

This package provides:
- Data loading and preprocessing (ml.data)
- Recommendation models (ml.models) 
- Evaluation metrics (ml.evaluation)
- Utility functions (ml.utils)
"""

__version__ = "0.1.0"
__author__ = "WhiskeyHub ML Team"

# Package-level imports will be added as components are built
# from .data import DataLoader, Preprocessor
# from .models import HybridRecommender, CollaborativeRecommender, ContentBasedRecommender
# from .evaluation import evaluate_model, cross_validate
# from .utils import calculate_similarity

# For now, expose version and basic info
__all__ = [
    "__version__",
    "__author__",
]

# Package configuration
DEFAULT_CONFIG = {
    "data_path": "../data/WhiskeyHubMySQL_6_13_2025_pt2/",
    "results_path": "../results/",
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
}

def get_config():
    """Get default package configuration."""
    return DEFAULT_CONFIG.copy()