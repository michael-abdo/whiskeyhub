"""
Evaluation metrics for WhiskeyHub recommendation system.

This module provides comprehensive evaluation metrics for recommendation models,
including rating prediction accuracy, ranking quality, and business metrics.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
import warnings

# Set up logging
logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """
    Comprehensive evaluator for recommendation systems.
    
    Provides rating prediction metrics (RMSE, MAE, R²) and ranking metrics
    (Precision@K, Recall@K, NDCG@K) along with business metrics like
    diversity and novelty.
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize recommendation evaluator.
        
        Args:
            k_values: List of K values for ranking metrics (default: [5, 10, 20])
        """
        self.k_values = k_values or [5, 10, 20]
        logger.info(f"Initialized RecommendationEvaluator with K values: {self.k_values}")
    
    def evaluate_model(
        self,
        model: Any,
        test_data: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'whiskey_id', 
        rating_col: str = 'rating'
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained recommendation model
            test_data: Test dataset
            user_col: Name of user column
            item_col: Name of item column
            rating_col: Name of rating column
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Starting comprehensive model evaluation...")
        
        results = {}
        
        # Rating prediction metrics
        rating_metrics = self.evaluate_rating_prediction(
            model, test_data, user_col, item_col, rating_col
        )
        results.update(rating_metrics)
        
        # Ranking metrics
        ranking_metrics = self.evaluate_ranking_quality(
            model, test_data, user_col, item_col, rating_col
        )
        results.update(ranking_metrics)
        
        # Business metrics
        business_metrics = self.evaluate_business_metrics(
            model, test_data, user_col, item_col, rating_col
        )
        results.update(business_metrics)
        
        logger.info(f"Evaluation completed with {len(results)} metrics")
        return results
    
    def evaluate_rating_prediction(
        self,
        model: Any,
        test_data: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'whiskey_id',
        rating_col: str = 'rating'
    ) -> Dict[str, float]:
        """
        Evaluate rating prediction accuracy.
        
        Args:
            model: Trained recommendation model
            test_data: Test dataset
            user_col: Name of user column
            item_col: Name of item column
            rating_col: Name of rating column
            
        Returns:
            Dictionary with rating prediction metrics
        """
        logger.info("Evaluating rating prediction accuracy...")
        
        # Get predictions for test data
        predictions = []
        actual_ratings = []
        
        for _, row in test_data.iterrows():
            user_id = row[user_col]
            item_id = row[item_col]
            actual_rating = row[rating_col]
            
            try:
                predicted_rating = model.predict_rating(user_id, item_id)
                predictions.append(predicted_rating)
                actual_ratings.append(actual_rating)
            except Exception as e:
                logger.debug(f"Prediction failed for user {user_id}, item {item_id}: {e}")
        
        if not predictions:
            logger.warning("No successful predictions made")
            return {}
        
        predictions = np.array(predictions)
        actual_ratings = np.array(actual_ratings)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
        mae = mean_absolute_error(actual_ratings, predictions)
        
        # R² score with error handling
        try:
            r2 = r2_score(actual_ratings, predictions)
        except:
            r2 = np.nan
        
        # Additional metrics
        mean_error = np.mean(predictions - actual_ratings)
        std_error = np.std(predictions - actual_ratings)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mean_error': mean_error,
            'std_error': std_error,
            'n_predictions': len(predictions)
        }
    
    def evaluate_ranking_quality(
        self,
        model: Any,
        test_data: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'whiskey_id',
        rating_col: str = 'rating',
        rating_threshold: float = 7.0
    ) -> Dict[str, float]:
        """
        Evaluate ranking quality using Precision@K, Recall@K, and NDCG@K.
        
        Args:
            model: Trained recommendation model
            test_data: Test dataset
            user_col: Name of user column
            item_col: Name of item column
            rating_col: Name of rating column
            rating_threshold: Threshold for considering items as relevant
            
        Returns:
            Dictionary with ranking metrics
        """
        logger.info("Evaluating ranking quality...")
        
        # Group test data by user
        user_groups = test_data.groupby(user_col)
        
        precision_scores = {k: [] for k in self.k_values}
        recall_scores = {k: [] for k in self.k_values}
        ndcg_scores = {k: [] for k in self.k_values}
        
        users_evaluated = 0
        
        for user_id, user_data in user_groups:
            # Get relevant items (highly rated items)
            relevant_items = set(
                user_data[user_data[rating_col] >= rating_threshold][item_col].tolist()
            )
            
            if not relevant_items:
                continue  # Skip users with no relevant items
            
            try:
                # Get recommendations for this user
                max_k = max(self.k_values)
                recommendations = model.recommend(user_id, n_recommendations=max_k)
                recommended_items = [item_id for item_id, _ in recommendations]
                
                # Calculate metrics for each K
                for k in self.k_values:
                    top_k_items = set(recommended_items[:k])
                    
                    # Precision@K
                    precision_k = len(top_k_items & relevant_items) / k if k > 0 else 0
                    precision_scores[k].append(precision_k)
                    
                    # Recall@K
                    recall_k = len(top_k_items & relevant_items) / len(relevant_items) if relevant_items else 0
                    recall_scores[k].append(recall_k)
                    
                    # NDCG@K
                    ndcg_k = self._calculate_ndcg_at_k(
                        recommended_items[:k], relevant_items, user_data, item_col, rating_col
                    )
                    ndcg_scores[k].append(ndcg_k)
                
                users_evaluated += 1
                
            except Exception as e:
                logger.debug(f"Ranking evaluation failed for user {user_id}: {e}")
        
        # Calculate average metrics
        results = {}
        for k in self.k_values:
            if precision_scores[k]:
                results[f'precision_at_{k}'] = np.mean(precision_scores[k])
                results[f'recall_at_{k}'] = np.mean(recall_scores[k])
                results[f'ndcg_at_{k}'] = np.mean(ndcg_scores[k])
        
        results['users_evaluated'] = users_evaluated
        return results
    
    def evaluate_business_metrics(
        self,
        model: Any,
        test_data: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'whiskey_id',
        rating_col: str = 'rating'
    ) -> Dict[str, float]:
        """
        Evaluate business-relevant metrics like diversity and novelty.
        
        Args:
            model: Trained recommendation model
            test_data: Test dataset
            user_col: Name of user column
            item_col: Name of item column
            rating_col: Name of rating column
            
        Returns:
            Dictionary with business metrics
        """
        logger.info("Evaluating business metrics...")
        
        # Get sample of users for efficiency
        unique_users = test_data[user_col].unique()
        sample_users = np.random.choice(
            unique_users, 
            size=min(50, len(unique_users)), 
            replace=False
        )
        
        all_recommendations = []
        coverage_items = set()
        
        for user_id in sample_users:
            try:
                recommendations = model.recommend(user_id, n_recommendations=10)
                user_items = [item_id for item_id, _ in recommendations]
                all_recommendations.extend(user_items)
                coverage_items.update(user_items)
                
            except Exception as e:
                logger.debug(f"Business metrics evaluation failed for user {user_id}: {e}")
        
        # Calculate metrics
        results = {}
        
        # Diversity (how varied are the recommendations)
        if all_recommendations:
            unique_items = set(all_recommendations)
            diversity = len(unique_items) / len(all_recommendations)
            results['diversity'] = diversity
        
        # Coverage (what fraction of items are ever recommended)
        total_items = test_data[item_col].nunique()
        if total_items > 0:
            coverage = len(coverage_items) / total_items
            results['coverage'] = coverage
        
        # Popularity bias (do we recommend popular items?)
        item_popularity = test_data.groupby(item_col).size()
        if all_recommendations and len(item_popularity) > 0:
            recommended_popularities = []
            for item_id in all_recommendations:
                if item_id in item_popularity:
                    recommended_popularities.append(item_popularity[item_id])
            
            if recommended_popularities:
                avg_popularity = np.mean(recommended_popularities)
                overall_avg_popularity = item_popularity.mean()
                popularity_bias = avg_popularity / overall_avg_popularity
                results['popularity_bias'] = popularity_bias
        
        results['sample_users'] = len(sample_users)
        return results
    
    def cross_validate_model(
        self,
        model_class: Any,
        data: pd.DataFrame,
        model_params: Dict = None,
        cv_folds: int = 5,
        user_col: str = 'user_id',
        item_col: str = 'whiskey_id',
        rating_col: str = 'rating',
        random_state: int = 42
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model_class: Recommendation model class
            data: Full dataset
            model_params: Parameters for model initialization
            cv_folds: Number of cross-validation folds
            user_col: Name of user column
            item_col: Name of item column
            rating_col: Name of rating column
            random_state: Random state for reproducible results
            
        Returns:
            Dictionary with lists of metric scores across folds
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation...")
        
        model_params = model_params or {}
        
        # Initialize KFold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        cv_results = {}
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            logger.info(f"Evaluating fold {fold + 1}/{cv_folds}")
            
            # Split data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            try:
                # Train model
                model = model_class(**model_params)
                model.fit(train_data, rating_col)
                
                # Evaluate model
                fold_results = self.evaluate_model(
                    model, test_data, user_col, item_col, rating_col
                )
                fold_metrics.append(fold_results)
                
            except Exception as e:
                logger.error(f"Fold {fold + 1} failed: {e}")
                continue
        
        if not fold_metrics:
            logger.error("All cross-validation folds failed")
            return {}
        
        # Aggregate results across folds
        for metric in fold_metrics[0].keys():
            values = [fold.get(metric, np.nan) for fold in fold_metrics]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                cv_results[f'{metric}_mean'] = np.mean(values)
                cv_results[f'{metric}_std'] = np.std(values)
                cv_results[f'{metric}_values'] = values
        
        cv_results['successful_folds'] = len(fold_metrics)
        
        logger.info(f"Cross-validation completed: {len(fold_metrics)}/{cv_folds} successful folds")
        return cv_results
    
    def _calculate_ndcg_at_k(
        self,
        recommended_items: List[Any],
        relevant_items: set,
        user_data: pd.DataFrame,
        item_col: str,
        rating_col: str
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K."""
        if not recommended_items:
            return 0.0
        
        # Create rating lookup for this user
        item_ratings = dict(zip(user_data[item_col], user_data[rating_col]))
        
        # Calculate DCG
        dcg = 0.0
        for i, item_id in enumerate(recommended_items):
            relevance = item_ratings.get(item_id, 0)
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_ratings = sorted([item_ratings.get(item_id, 0) for item_id in relevant_items], reverse=True)
        idcg = 0.0
        for i, rating in enumerate(ideal_ratings[:len(recommended_items)]):
            idcg += rating / np.log2(i + 2)
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0


# Convenience functions
def evaluate_model_performance(
    model: Any,
    test_data: pd.DataFrame,
    k_values: List[int] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Convenience function to evaluate a recommendation model.
    
    Args:
        model: Trained recommendation model
        test_data: Test dataset
        k_values: List of K values for ranking metrics
        **kwargs: Additional arguments for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = RecommendationEvaluator(k_values=k_values)
    return evaluator.evaluate_model(model, test_data, **kwargs)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_precision_at_k(recommended_items: List[Any], relevant_items: set, k: int) -> float:
    """Calculate Precision@K."""
    if k == 0:
        return 0.0
    
    top_k = set(recommended_items[:k])
    return len(top_k & relevant_items) / k


def calculate_recall_at_k(recommended_items: List[Any], relevant_items: set, k: int) -> float:
    """Calculate Recall@K."""
    if not relevant_items:
        return 0.0
    
    top_k = set(recommended_items[:k])
    return len(top_k & relevant_items) / len(relevant_items)


def compare_models(
    models: Dict[str, Any],
    test_data: pd.DataFrame,
    metrics: List[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Compare multiple recommendation models.
    
    Args:
        models: Dictionary mapping model names to model instances
        test_data: Test dataset
        metrics: List of metrics to compare (if None, uses all)
        **kwargs: Additional arguments for evaluation
        
    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Comparing {len(models)} models...")
    
    evaluator = RecommendationEvaluator()
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        try:
            model_results = evaluator.evaluate_model(model, test_data, **kwargs)
            results[model_name] = model_results
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
            results[model_name] = {}
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(results).T
    
    # Filter metrics if specified
    if metrics:
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        comparison_df = comparison_df[available_metrics]
    
    return comparison_df


class ABTester:
    """
    A/B testing framework for recommendation model optimization.
    
    Supports statistical significance testing and confidence intervals
    for comparing model performance.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize A/B tester.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        logger.info(f"Initialized ABTester with alpha={alpha}")
    
    def test_models(
        self,
        model_a: Any,
        model_b: Any,
        test_data: pd.DataFrame,
        metric: str = 'rmse',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform A/B test between two models.
        
        Args:
            model_a: First model (baseline)
            model_b: Second model (variant)
            test_data: Test dataset
            metric: Metric to compare
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary with A/B test results
        """
        logger.info(f"Running A/B test on metric: {metric}")
        
        evaluator = RecommendationEvaluator()
        
        # Evaluate both models
        results_a = evaluator.evaluate_model(model_a, test_data, **kwargs)
        results_b = evaluator.evaluate_model(model_b, test_data, **kwargs)
        
        if metric not in results_a or metric not in results_b:
            raise ValueError(f"Metric {metric} not available in evaluation results")
        
        # Statistical test (simplified - for production, use proper statistical tests)
        score_a = results_a[metric]
        score_b = results_b[metric]
        
        # For demonstration purposes - in practice, you'd need paired samples
        # and proper statistical testing (t-test, Wilcoxon, etc.)
        improvement = (score_b - score_a) / score_a if score_a != 0 else 0
        
        return {
            'model_a_score': score_a,
            'model_b_score': score_b,
            'improvement': improvement,
            'better_model': 'B' if score_b > score_a else 'A',
            'metric': metric,
            'results_a': results_a,
            'results_b': results_b
        }