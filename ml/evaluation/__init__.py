"""
Model evaluation and testing module for WhiskeyHub ML pipeline.

This module provides:
- Standard recommendation metrics (RMSE, MAE, Precision@K, Recall@K)
- Cross-validation framework
- A/B testing infrastructure  
- Business metrics (diversity, novelty)
- Statistical significance testing
"""

from .metrics import (
    RecommendationEvaluator,
    ABTester,
    evaluate_model_performance,
    calculate_rmse,
    calculate_precision_at_k,
    calculate_recall_at_k,
    compare_models
)

__all__ = [
    "RecommendationEvaluator",
    "ABTester", 
    "evaluate_model_performance",
    "calculate_rmse",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "compare_models",
]