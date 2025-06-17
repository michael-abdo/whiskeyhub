"""
Recommendation models module for WhiskeyHub ML pipeline.

This module contains:
- Base model abstract classes
- Linear baseline model (from existing analysis)
- Content-based filtering model 
- Collaborative filtering model
- Hybrid recommendation model
"""

from .base_model import BaseRecommender, BasePredictiveModel, BaseCollaborativeModel
from .linear_baseline import LinearBaselineModel, train_linear_baseline
# from .content_based import ContentBasedRecommender
# from .collaborative import CollaborativeRecommender  
# from .hybrid import HybridRecommender

__all__ = [
    "BaseRecommender",
    "BasePredictiveModel", 
    "BaseCollaborativeModel",
    "LinearBaselineModel",
    "train_linear_baseline",
    # "ContentBasedRecommender",
    # "CollaborativeRecommender",
    # "HybridRecommender",
]