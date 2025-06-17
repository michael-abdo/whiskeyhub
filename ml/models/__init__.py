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
from .content_based import ContentBasedRecommender, train_content_based_recommender
from .collaborative import CollaborativeRecommender, train_collaborative_recommender
from .hybrid import HybridRecommender, train_hybrid_recommender

__all__ = [
    "BaseRecommender",
    "BasePredictiveModel", 
    "BaseCollaborativeModel",
    "LinearBaselineModel",
    "train_linear_baseline",
    "ContentBasedRecommender",
    "train_content_based_recommender",
    "CollaborativeRecommender",
    "train_collaborative_recommender",
    "HybridRecommender",
    "train_hybrid_recommender",
]