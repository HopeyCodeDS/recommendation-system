"""
Abstract base class for recommender systems
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


class BaseRecommender(ABC):
    """
    Abstract base class for all recommender systems.
    Defines the common interface that all recommenders must implement.
    """
    
    def __init__(self, name: str = "BaseRecommender"):
        """
        Initialize the base recommender.
        
        Parameters:
        -----------
        name : str
            Name of the recommender
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame) -> 'BaseRecommender':
        """
        Train the recommender on the given ratings data.
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings dataframe with columns: user_id, item_id (or title), rating
            
        Returns:
        --------
        self
        """
        self.is_fitted = True
        return self
    
    @abstractmethod
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.
        
        Parameters:
        -----------
        user_id : str
            User identifier
        item_id : str
            Item identifier (book_id or title)
            
        Returns:
        --------
        float
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted before prediction")
        pass
    
    @abstractmethod
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_rated: bool = True) -> List[Dict]:
        """
        Get top N recommendations for a user.
        
        Parameters:
        -----------
        user_id : str
            User identifier
        n_recommendations : int
            Number of recommendations to return
        exclude_rated : bool
            Whether to exclude items the user has already rated
            
        Returns:
        --------
        List[Dict]
            List of recommendation dictionaries with keys:
            - item_id: Item identifier
            - predicted_rating: Predicted rating
            - confidence: Confidence score (optional)
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted before recommendation")
        pass
    
    def evaluate(self, test_ratings_df: pd.DataFrame,
                metrics: List[str] = ['rmse', 'mae']) -> Dict[str, float]:
        """
        Evaluate the recommender on test data.
        
        Parameters:
        -----------
        test_ratings_df : pd.DataFrame
            Test ratings dataframe
        metrics : List[str]
            List of metrics to compute
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of metric names and values
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted before evaluation")
        
        predictions = []
        actuals = []
        
        for _, row in test_ratings_df.iterrows():
            try:
                user_id = row['user_id']
                item_id = row.get('book_id') or row.get('title')
                actual_rating = row['rating']
                
                predicted_rating = self.predict(user_id, item_id)
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except (KeyError, ValueError):
                continue
        
        if len(predictions) == 0:
            return {metric: np.nan for metric in metrics}
        
        results = {}
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        
        if 'mae' in metrics:
            results['mae'] = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        
        return results
    
    def get_user_rated_items(self, user_id: str, ratings_df: pd.DataFrame) -> set:
        """
        Get set of items rated by a user.
        
        Parameters:
        -----------
        user_id : str
            User identifier
        ratings_df : pd.DataFrame
            Ratings dataframe
            
        Returns:
        --------
        set
            Set of item identifiers
        """
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        item_col = 'book_id' if 'book_id' in user_ratings.columns else 'title'
        return set(user_ratings[item_col].unique())
    
    def __repr__(self) -> str:
        """String representation of the recommender."""
        return f"{self.name}(fitted={self.is_fitted})"

