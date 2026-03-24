"""
Evaluation framework for recommender systems
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import warnings

from .metrics import (
    precision_at_k, recall_at_k, f1_at_k,
    rmse, mae, coverage, diversity, novelty
)
from ..recommenders.base_recommender import BaseRecommender


class RecommenderEvaluator:
    """
    Comprehensive evaluator for recommender systems.
    """
    
    def __init__(self,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 k_values: List[int] = [5, 10, 20],
                 min_rating_threshold: float = 3.0):
        """
        Initialize evaluator.
        
        Parameters:
        -----------
        test_size : float
            Fraction of data to use for testing
        random_state : int
            Random seed for reproducibility
        k_values : List[int]
            List of K values for precision/recall@K
        min_rating_threshold : float
            Minimum rating to consider as "relevant"
        """
        self.test_size = test_size
        self.random_state = random_state
        self.k_values = k_values
        self.min_rating_threshold = min_rating_threshold
        
        self.train_data = None
        self.test_data = None
    
    def train_test_split(self, ratings_df: pd.DataFrame,
                        split_method: str = 'random') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings dataframe
        split_method : str
            'random' or 'temporal' (if timestamp available)
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Train and test dataframes
        """
        if split_method == 'temporal' and 'timestamp' in ratings_df.columns:
            # Temporal split: use earlier data for training
            ratings_df_sorted = ratings_df.sort_values('timestamp')
            split_idx = int(len(ratings_df_sorted) * (1 - self.test_size))
            self.train_data = ratings_df_sorted.iloc[:split_idx].copy()
            self.test_data = ratings_df_sorted.iloc[split_idx:].copy()
        else:
            # Random split
            self.train_data, self.test_data = train_test_split(
                ratings_df,
                test_size=self.test_size,
                random_state=self.random_state
            )
        
        return self.train_data, self.test_data
    
    def evaluate(self,
                recommender: BaseRecommender,
                test_data: Optional[pd.DataFrame] = None,
                train_data: Optional[pd.DataFrame] = None,
                metrics: List[str] = ['precision', 'recall', 'rmse', 'mae']) -> Dict:
        """
        Evaluate recommender on test data.
        
        Parameters:
        -----------
        recommender : BaseRecommender
            Trained recommender to evaluate
        test_data : pd.DataFrame, optional
            Test data (if None, uses self.test_data)
        train_data : pd.DataFrame, optional
            Train data (for getting user rated items)
        metrics : List[str]
            Metrics to compute
            
        Returns:
        --------
        Dict
            Dictionary of evaluation results
        """
        if test_data is None:
            test_data = self.test_data
        
        if test_data is None:
            raise ValueError("Test data must be provided or set via train_test_split")
        
        if train_data is None:
            train_data = self.train_data
        
        results = {}
        
        # Rating prediction metrics
        if 'rmse' in metrics or 'mae' in metrics:
            predictions = []
            actuals = []
            
            for _, row in test_data.iterrows():
                try:
                    user_id = row['user_id']
                    item_id = row.get('book_id') or row.get('title')
                    actual_rating = row['rating']
                    
                    predicted_rating = recommender.predict(user_id, item_id)
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
                except Exception as e:
                    continue
            
            if len(predictions) > 0:
                if 'rmse' in metrics:
                    results['rmse'] = rmse(predictions, actuals)
                if 'mae' in metrics:
                    results['mae'] = mae(predictions, actuals)
        
        # Ranking metrics (precision, recall, F1)
        if any(m in metrics for m in ['precision', 'recall', 'f1']):
            precision_scores = {f'precision@{k}': [] for k in self.k_values}
            recall_scores = {f'recall@{k}': [] for k in self.k_values}
            f1_scores = {f'f1@{k}': [] for k in self.k_values}
            
            # Group test data by user
            test_by_user = test_data.groupby('user_id')
            
            for user_id, user_test_data in test_by_user:
                try:
                    # Get relevant items (items with rating >= threshold)
                    relevant_items = set(
                        user_test_data[
                            user_test_data['rating'] >= self.min_rating_threshold
                        ][user_test_data.columns[user_test_data.columns.isin(['book_id', 'title'])][0]]
                    )
                    
                    if len(relevant_items) == 0:
                        continue
                    
                    # Get recommendations
                    recommendations = recommender.recommend(
                        user_id,
                        n_recommendations=max(self.k_values),
                        exclude_rated=True
                    )
                    
                    recommended_items = [rec['item_id'] for rec in recommendations]
                    
                    # Compute metrics for each k
                    for k in self.k_values:
                        if 'precision' in metrics:
                            prec = precision_at_k(recommended_items, relevant_items, k)
                            precision_scores[f'precision@{k}'].append(prec)
                        
                        if 'recall' in metrics:
                            rec = recall_at_k(recommended_items, relevant_items, k)
                            recall_scores[f'recall@{k}'].append(rec)
                        
                        if 'f1' in metrics:
                            f1 = f1_at_k(recommended_items, relevant_items, k)
                            f1_scores[f'f1@{k}'].append(f1)
                
                except Exception as e:
                    continue
            
            # Average scores
            for k in self.k_values:
                if 'precision' in metrics:
                    results[f'precision@{k}'] = np.mean(precision_scores[f'precision@{k}']) if precision_scores[f'precision@{k}'] else 0.0
                if 'recall' in metrics:
                    results[f'recall@{k}'] = np.mean(recall_scores[f'recall@{k}']) if recall_scores[f'recall@{k}'] else 0.0
                if 'f1' in metrics:
                    results[f'f1@{k}'] = np.mean(f1_scores[f'f1@{k}']) if f1_scores[f'f1@{k}'] else 0.0
        
        # Coverage
        if 'coverage' in metrics:
            all_items = set(test_data.get('book_id', test_data.get('title', [])))
            all_recommendations = []
            
            test_by_user = test_data.groupby('user_id')
            for user_id, _ in test_by_user:
                try:
                    recommendations = recommender.recommend(user_id, n_recommendations=20)
                    recommended_items = [rec['item_id'] for rec in recommendations]
                    all_recommendations.append(recommended_items)
                except:
                    continue
            
            results['coverage'] = coverage(all_recommendations, all_items)
        
        return results
    
    def compare_recommenders(self,
                           recommenders: Dict[str, BaseRecommender],
                           test_data: Optional[pd.DataFrame] = None,
                           metrics: List[str] = ['precision', 'recall', 'rmse', 'mae']) -> pd.DataFrame:
        """
        Compare multiple recommenders.
        
        Parameters:
        -----------
        recommenders : Dict[str, BaseRecommender]
            Dictionary mapping recommender names to instances
        test_data : pd.DataFrame, optional
            Test data
        metrics : List[str]
            Metrics to compute
            
        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        comparison_results = []
        
        for name, recommender in recommenders.items():
            print(f"Evaluating {name}...")
            results = self.evaluate(recommender, test_data=test_data, metrics=metrics)
            results['recommender'] = name
            comparison_results.append(results)
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.set_index('recommender')
        
        return comparison_df
    
    def print_results(self, results: Dict):
        """
        Print evaluation results in a formatted way.
        
        Parameters:
        -----------
        results : Dict
            Evaluation results dictionary
        """
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        # Rating prediction metrics
        if 'rmse' in results:
            print(f"\nRMSE: {results['rmse']:.4f}")
        if 'mae' in results:
            print(f"MAE: {results['mae']:.4f}")
        
        # Ranking metrics
        precision_keys = [k for k in results.keys() if k.startswith('precision@')]
        if precision_keys:
            print("\nPrecision@K:")
            for key in sorted(precision_keys):
                print(f"  {key}: {results[key]:.4f}")
        
        recall_keys = [k for k in results.keys() if k.startswith('recall@')]
        if recall_keys:
            print("\nRecall@K:")
            for key in sorted(recall_keys):
                print(f"  {key}: {results[key]:.4f}")
        
        f1_keys = [k for k in results.keys() if k.startswith('f1@')]
        if f1_keys:
            print("\nF1@K:")
            for key in sorted(f1_keys):
                print(f"  {key}: {results[key]:.4f}")
        
        if 'coverage' in results:
            print(f"\nCoverage: {results['coverage']:.4f}")
        
        print("=" * 60)

