"""
Hybrid Recommender System
Combines collaborative filtering and content-based filtering
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal
import warnings

from .base_recommender import BaseRecommender
from .collaborative_filter import CollaborativeFilter
from .content_based import ContentBasedRecommender


class HybridRecommender(BaseRecommender):
    """
    Hybrid Recommender that combines collaborative filtering and content-based filtering.
    Uses adaptive weighting based on user history.
    """
    
    def __init__(self,
                 cf_weight: float = 0.6,
                 cb_weight: float = 0.4,
                 adaptive_weighting: bool = True,
                 min_user_ratings_for_cf: int = 10,
                 cf_method: Literal['user-user', 'item-item'] = 'user-user',
                 cf_k_neighbors: int = 50,
                 cb_max_features: int = 5000,
                 cb_n_neighbors: int = 10):
        """
        Initialize hybrid recommender.
        
        Parameters:
        -----------
        cf_weight : float
            Weight for collaborative filtering (0-1)
        cb_weight : float
            Weight for content-based filtering (0-1)
        adaptive_weighting : bool
            Whether to adaptively adjust weights based on user history
        min_user_ratings_for_cf : int
            Minimum ratings for user to use higher CF weight
        cf_method : str
            Collaborative filtering method: 'user-user' or 'item-item'
        cf_k_neighbors : int
            Number of neighbors for collaborative filtering
        cb_max_features : int
            Maximum features for content-based TF-IDF
        cb_n_neighbors : int
            Number of neighbors for content-based
        """
        super().__init__(name="HybridRecommender")
        
        # Validate weights
        if abs(cf_weight + cb_weight - 1.0) > 0.01:
            warnings.warn("Weights should sum to 1.0. Normalizing...")
            total = cf_weight + cb_weight
            cf_weight = cf_weight / total
            cb_weight = cb_weight / total
        
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.adaptive_weighting = adaptive_weighting
        self.min_user_ratings_for_cf = min_user_ratings_for_cf
        
        # Initialize component recommenders
        self.cf_recommender = CollaborativeFilter(
            method=cf_method,
            k_neighbors=cf_k_neighbors,
            min_user_ratings=1,  # More lenient for small datasets
            min_item_ratings=1
        )
        
        self.cb_recommender = ContentBasedRecommender(
            max_features=cb_max_features,
            n_neighbors=cb_n_neighbors
        )
        
        # Store ratings for adaptive weighting
        self.ratings_df = None
    
    def fit(self, ratings_df: pd.DataFrame, books_df: Optional[pd.DataFrame] = None) -> 'HybridRecommender':
        """
        Train both component recommenders.
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings dataframe
        books_df : pd.DataFrame, optional
            Books metadata dataframe
            
        Returns:
        --------
        self
        """
        self.ratings_df = ratings_df.copy()
        
        # Fit collaborative filter
        print("Fitting collaborative filter...")
        self.cf_recommender.fit(ratings_df)
        
        # Fit content-based recommender
        print("Fitting content-based recommender...")
        self.cb_recommender.fit(ratings_df, books_df)
        
        return super().fit(ratings_df)
    
    def _get_adaptive_weights(self, user_id: str) -> tuple:
        """
        Get adaptive weights for a user based on their rating history.
        
        Parameters:
        -----------
        user_id : str
            User identifier
            
        Returns:
        --------
        tuple
            (cf_weight, cb_weight) adjusted weights
        """
        if not self.adaptive_weighting:
            return self.cf_weight, self.cb_weight
        
        # Count user's ratings
        if self.ratings_df is not None:
            user_ratings_count = len(self.ratings_df[self.ratings_df['user_id'] == user_id])
        else:
            user_ratings_count = 0
        
        # Adjust weights based on user history
        if user_ratings_count >= self.min_user_ratings_for_cf:
            # User has enough ratings - favor collaborative filtering
            cf_w = 0.7
            cb_w = 0.3
        elif user_ratings_count > 0:
            # User has some ratings - balanced approach
            # Interpolate based on rating count
            ratio = user_ratings_count / self.min_user_ratings_for_cf
            cf_w = 0.3 + (0.4 * ratio)  # 0.3 to 0.7
            cb_w = 1.0 - cf_w
        else:
            # New user - favor content-based
            cf_w = 0.2
            cb_w = 0.8
        
        return cf_w, cb_w
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating using weighted combination of CF and CB.
        
        Parameters:
        -----------
        user_id : str
            User identifier
        item_id : str
            Item identifier
            
        Returns:
        --------
        float
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get adaptive weights
        cf_w, cb_w = self._get_adaptive_weights(user_id)
        
        # Get predictions from both recommenders
        try:
            cf_prediction = self.cf_recommender.predict(user_id, item_id)
        except Exception as e:
            warnings.warn(f"CF prediction failed: {e}. Using fallback.")
            cf_prediction = 3.0
        
        try:
            cb_prediction = self.cb_recommender.predict(user_id, item_id)
        except Exception as e:
            warnings.warn(f"CB prediction failed: {e}. Using fallback.")
            cb_prediction = 3.0
        
        # Weighted combination
        prediction = (cf_w * cf_prediction) + (cb_w * cb_prediction)
        
        # Clamp to valid range
        return np.clip(prediction, 1.0, 5.0)
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_rated: bool = True) -> List[Dict]:
        """
        Get top N recommendations using hybrid approach.
        
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
            List of recommendation dictionaries
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommendation")
        
        # Get adaptive weights
        cf_w, cb_w = self._get_adaptive_weights(user_id)
        
        # Get user's rated items for content-based
        user_rated_items = None
        if exclude_rated and self.ratings_df is not None:
            user_rated_items = self.get_user_rated_items(user_id, self.ratings_df)
            user_rated_items = list(user_rated_items)
        
        # Get recommendations from both recommenders
        try:
            cf_recommendations = self.cf_recommender.recommend(
                user_id,
                n_recommendations=n_recommendations * 2,  # Get more for blending
                exclude_rated=exclude_rated
            )
        except Exception as e:
            warnings.warn(f"CF recommendations failed: {e}")
            cf_recommendations = []
        
        try:
            cb_recommendations = self.cb_recommender.recommend(
                user_id,
                n_recommendations=n_recommendations * 2,
                exclude_rated=exclude_rated,
                user_rated_items=user_rated_items
            )
        except Exception as e:
            warnings.warn(f"CB recommendations failed: {e}")
            cb_recommendations = []
        
        # Combine recommendations
        combined_scores = {}
        
        # Add CF recommendations
        for rec in cf_recommendations:
            item_id = rec['item_id']
            score = rec.get('predicted_rating', 3.0) * cf_w
            confidence = rec.get('confidence', 0.5) * cf_w
            
            if item_id not in combined_scores:
                combined_scores[item_id] = {
                    'item_id': item_id,
                    'score': 0.0,
                    'confidence': 0.0,
                    'cf_score': 0.0,
                    'cb_score': 0.0
                }
            
            combined_scores[item_id]['score'] += score
            combined_scores[item_id]['confidence'] += confidence
            combined_scores[item_id]['cf_score'] = rec.get('predicted_rating', 3.0)
        
        # Add CB recommendations
        for rec in cb_recommendations:
            item_id = rec['item_id']
            score = rec.get('predicted_rating', 3.0) * cb_w
            confidence = rec.get('confidence', 0.5) * cb_w
            
            if item_id not in combined_scores:
                combined_scores[item_id] = {
                    'item_id': item_id,
                    'score': 0.0,
                    'confidence': 0.0,
                    'cf_score': 0.0,
                    'cb_score': 0.0
                }
            
            combined_scores[item_id]['score'] += score
            combined_scores[item_id]['confidence'] += confidence
            combined_scores[item_id]['cb_score'] = rec.get('predicted_rating', 3.0)
        
        # Convert to list and sort
        recommendations = []
        for item_id, data in combined_scores.items():
            recommendations.append({
                'item_id': item_id,
                'predicted_rating': data['score'],
                'confidence': min(1.0, data['confidence']),
                'cf_rating': data.get('cf_score', 0.0),
                'cb_rating': data.get('cb_score', 0.0)
            })
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations[:n_recommendations]

