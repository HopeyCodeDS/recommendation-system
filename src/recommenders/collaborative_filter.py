"""
Collaborative Filtering Recommender
Implements user-user and item-item collaborative filtering
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, issparse
from typing import List, Dict, Optional, Literal
import warnings

from .base_recommender import BaseRecommender
from ..utils.similarity import cosine_similarity_sparse, apply_similarity_threshold, normalize_similarity_matrix
from ..utils.preprocessing import normalize_ratings, create_user_item_matrix


class CollaborativeFilter(BaseRecommender):
    """
    Collaborative Filtering Recommender using user-user similarity.
    Supports both user-user and item-item collaborative filtering.
    """
    
    def __init__(self, 
                 method: Literal['user-user', 'item-item'] = 'user-user',
                 similarity_metric: str = 'cosine',
                 k_neighbors: int = 50,
                 min_similarity: float = 0.0,
                 normalize_ratings: bool = True,
                 min_user_ratings: int = 3,
                 min_item_ratings: int = 3):
        """
        Initialize collaborative filter.
        
        Parameters:
        -----------
        method : str
            'user-user' or 'item-item' collaborative filtering
        similarity_metric : str
            Similarity metric: 'cosine' or 'pearson'
        k_neighbors : int
            Number of nearest neighbors to consider
        min_similarity : float
            Minimum similarity threshold
        normalize_ratings : bool
            Whether to mean-center ratings
        min_user_ratings : int
            Minimum ratings per user
        min_item_ratings : int
            Minimum ratings per item
        """
        super().__init__(name=f"CollaborativeFilter({method})")
        self.method = method
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self.normalize_ratings_flag = normalize_ratings
        self.min_user_ratings = min_user_ratings
        self.min_item_ratings = min_item_ratings
        
        # Model state
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None
        self.ratings_df = None
        self.item_col = None
    
    def fit(self, ratings_df: pd.DataFrame) -> 'CollaborativeFilter':
        """
        Train the collaborative filter.
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings dataframe with columns: user_id, item_id (or title), rating
            
        Returns:
        --------
        self
        """
        self.ratings_df = ratings_df.copy()
        
        # Determine item column
        self.item_col = 'book_id' if 'book_id' in ratings_df.columns else 'title'
        
        # Filter sparse data
        from ..utils.preprocessing import filter_sparse_data
        filtered_df = filter_sparse_data(
            self.ratings_df,
            min_user_ratings=self.min_user_ratings,
            min_item_ratings=self.min_item_ratings
        )
        
        # Check if we have enough data after filtering
        if len(filtered_df) == 0:
            raise ValueError("No data remaining after filtering. Try reducing min_user_ratings or min_item_ratings.")
        
        unique_users = filtered_df['user_id'].nunique()
        unique_items = filtered_df[self.item_col].nunique()
        
        if unique_users < 2:
            raise ValueError(f"Need at least 2 users, but only found {unique_users} after filtering.")
        
        if unique_items < 2:
            raise ValueError(f"Need at least 2 items, but only found {unique_items} after filtering.")
        
        # Normalize ratings if requested
        if self.normalize_ratings_flag:
            filtered_df = normalize_ratings(filtered_df, method='mean_centering')
            rating_col = 'normalized_rating'
        else:
            rating_col = 'rating'
        
        # Calculate means
        self.user_means = filtered_df.groupby('user_id')['rating'].mean()
        self.item_means = filtered_df.groupby(self.item_col)['rating'].mean()
        self.global_mean = filtered_df['rating'].mean()
        
        # Create user-item matrix
        self.user_item_matrix, self.user_mapping, self.item_mapping = create_user_item_matrix(
            filtered_df,
            item_col=self.item_col
        )
        
        # Create reverse mappings
        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
        
        # Compute similarity matrix
        if self.method == 'user-user':
            self._compute_user_similarity()
        else:
            self._compute_item_similarity()
        
        return super().fit(ratings_df)
    
    def _compute_user_similarity(self):
        """Compute user-user similarity matrix."""
        # Convert to dense for similarity computation
        dense_matrix = self.user_item_matrix.toarray()
        
        if self.similarity_metric == 'cosine':
            self.similarity_matrix = cosine_similarity_sparse(self.user_item_matrix, dense_output=True)
        else:
            # Pearson correlation
            from ..utils.similarity import pearson_correlation
            self.similarity_matrix = pearson_correlation(dense_matrix)
        
        # Apply threshold
        if self.min_similarity > 0:
            self.similarity_matrix = apply_similarity_threshold(
                self.similarity_matrix,
                threshold=self.min_similarity
            )
        
        # Normalize
        self.similarity_matrix = normalize_similarity_matrix(self.similarity_matrix)
    
    def _compute_item_similarity(self):
        """Compute item-item similarity matrix."""
        # Transpose matrix for item-item similarity
        item_user_matrix = self.user_item_matrix.T
        dense_matrix = item_user_matrix.toarray()
        
        if self.similarity_metric == 'cosine':
            self.similarity_matrix = cosine_similarity_sparse(item_user_matrix, dense_output=True)
        else:
            from ..utils.similarity import pearson_correlation
            self.similarity_matrix = pearson_correlation(dense_matrix)
        
        # Apply threshold
        if self.min_similarity > 0:
            self.similarity_matrix = apply_similarity_threshold(
                self.similarity_matrix,
                threshold=self.min_similarity
            )
        
        # Normalize
        self.similarity_matrix = normalize_similarity_matrix(self.similarity_matrix)
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.
        
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
        
        # Handle cold-start
        if user_id not in self.user_mapping:
            # Return item mean or global mean
            if item_id in self.item_mapping:
                return self.item_means.get(item_id, self.global_mean)
            return self.global_mean
        
        if item_id not in self.item_mapping:
            # Return user mean or global mean
            return self.user_means.get(user_id, self.global_mean)
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        if self.method == 'user-user':
            return self._predict_user_user(user_idx, item_idx, user_id)
        else:
            return self._predict_item_item(user_idx, item_idx, item_id)
    
    def _predict_user_user(self, user_idx: int, item_idx: int, user_id: str) -> float:
        """Predict using user-user collaborative filtering."""
        # Get user's ratings
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Check if user has rated this item
        if user_ratings[item_idx] != 0:
            # Return actual rating (adjusted if normalized)
            rating = user_ratings[item_idx]
            if self.normalize_ratings_flag:
                return rating + self.user_means[user_id]
            return rating
        
        # Get similar users
        user_similarities = self.similarity_matrix[user_idx]
        
        # Get k nearest neighbors
        top_k_indices = np.argsort(user_similarities)[::-1][:self.k_neighbors + 1]
        top_k_indices = top_k_indices[top_k_indices != user_idx]  # Exclude self
        top_k_indices = top_k_indices[:self.k_neighbors]
        
        if len(top_k_indices) == 0:
            # Fallback to user mean or global mean
            return self.user_means.get(user_id, self.global_mean)
        
        # Get ratings from similar users for this item
        similar_users_ratings = self.user_item_matrix[top_k_indices, item_idx].toarray().flatten()
        similar_users_sims = user_similarities[top_k_indices]
        
        # Filter out zero ratings (unrated items)
        mask = similar_users_ratings != 0
        if not mask.any():
            return self.user_means.get(user_id, self.global_mean)
        
        similar_users_ratings = similar_users_ratings[mask]
        similar_users_sims = similar_users_sims[mask]
        
        # Weighted average
        if self.normalize_ratings_flag:
            # Add back means for similar users
            similar_user_ids = [self.reverse_user_mapping[idx] for idx in top_k_indices[mask]]
            similar_user_means = np.array([self.user_means.get(uid, 0) for uid in similar_user_ids])
            similar_users_ratings = similar_users_ratings + similar_user_means
        
        # Compute weighted prediction
        if np.sum(np.abs(similar_users_sims)) > 0:
            prediction = np.sum(similar_users_ratings * similar_users_sims) / np.sum(np.abs(similar_users_sims))
        else:
            prediction = self.user_means.get(user_id, self.global_mean)
        
        # Add user mean if normalized
        if self.normalize_ratings_flag:
            prediction = prediction + self.user_means[user_id]
        
        # Clamp to valid range
        return np.clip(prediction, 1.0, 5.0)
    
    def _predict_item_item(self, item_idx: int, user_idx: int, item_id: str) -> float:
        """Predict using item-item collaborative filtering."""
        # Get item's ratings (across all users) - this is a column vector
        item_ratings = self.user_item_matrix[:, item_idx].toarray().flatten()
        
        # Check if user has rated this item
        if user_idx < len(item_ratings) and item_ratings[user_idx] != 0:
            rating = item_ratings[user_idx]
            if self.normalize_ratings_flag:
                return rating + self.user_means.get(self.reverse_user_mapping[user_idx], self.global_mean)
            return rating
        
        # Get similar items
        item_similarities = self.similarity_matrix[item_idx]
        
        # Get k nearest neighbors
        top_k_indices = np.argsort(item_similarities)[::-1][:self.k_neighbors + 1]
        top_k_indices = top_k_indices[top_k_indices != item_idx]  # Exclude self
        top_k_indices = top_k_indices[:self.k_neighbors]
        
        if len(top_k_indices) == 0:
            return self.item_means.get(item_id, self.global_mean)
        
        # Get user's ratings for similar items
        # In item-item mode, user_item_matrix is (users x items)
        # We need to get the user's row and then extract ratings for similar items
        if user_idx < self.user_item_matrix.shape[0]:
            user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
            similar_items_ratings = user_ratings[top_k_indices]
            similar_items_sims = item_similarities[top_k_indices]
        else:
            # User index out of bounds - return item mean
            return self.item_means.get(item_id, self.global_mean)
        
        # Filter out zero ratings
        mask = similar_items_ratings != 0
        if not mask.any():
            return self.item_means.get(item_id, self.global_mean)
        
        similar_items_ratings = similar_items_ratings[mask]
        similar_items_sims = similar_items_sims[mask]
        
        # Weighted average
        if self.normalize_ratings_flag:
            # Add back item means
            similar_item_ids = [self.reverse_item_mapping[idx] for idx in top_k_indices[mask]]
            similar_item_means = np.array([self.item_means.get(iid, 0) for iid in similar_item_ids])
            similar_items_ratings = similar_items_ratings + similar_item_means
        
        # Compute weighted prediction
        if np.sum(np.abs(similar_items_sims)) > 0:
            prediction = np.sum(similar_items_ratings * similar_items_sims) / np.sum(np.abs(similar_items_sims))
        else:
            prediction = self.item_means.get(item_id, self.global_mean)
        
        # Add item mean if normalized
        if self.normalize_ratings_flag:
            prediction = prediction + self.item_means[item_id]
        
        # Clamp to valid range
        return np.clip(prediction, 1.0, 5.0)
    
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
            List of recommendation dictionaries
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommendation")
        
        # Handle cold-start
        if user_id not in self.user_mapping:
            return self._get_popular_recommendations(n_recommendations)
        
        user_idx = self.user_mapping[user_id]
        
        # Get user's rated items
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Predict ratings for all items
        predictions = []
        for item_idx in range(len(self.item_mapping)):
            item_id = self.reverse_item_mapping[item_idx]
            
            # Skip if user has already rated and exclude_rated is True
            if exclude_rated and user_ratings[item_idx] != 0:
                continue
            
            predicted_rating = self.predict(user_id, item_id)
            predictions.append({
                'item_id': item_id,
                'predicted_rating': predicted_rating,
                'confidence': abs(user_ratings[item_idx]) if user_ratings[item_idx] != 0 else 0.5
            })
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return predictions[:n_recommendations]
    
    def _get_popular_recommendations(self, n_recommendations: int) -> List[Dict]:
        """Get popular items as fallback for cold-start users."""
        # Sort items by mean rating and number of ratings
        item_stats = self.ratings_df.groupby(self.item_col).agg({
            'rating': ['mean', 'count']
        })
        item_stats.columns = ['mean_rating', 'count']
        item_stats = item_stats.sort_values(['mean_rating', 'count'], ascending=False)
        
        recommendations = []
        for item_id in item_stats.head(n_recommendations).index:
            recommendations.append({
                'item_id': item_id,
                'predicted_rating': item_stats.loc[item_id, 'mean_rating'],
                'confidence': 0.3  # Low confidence for cold-start
            })
        
        return recommendations

