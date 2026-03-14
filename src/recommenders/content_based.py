"""
Content-Based Recommender
Uses TF-IDF vectorization on book metadata (title, authors, tags)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional
import warnings

from .base_recommender import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """
    Content-Based Recommender using TF-IDF on book metadata.
    """
    
    def __init__(self,
                 max_features: int = 5000,
                 min_df: int = 1,
                 max_df: float = 0.8,
                 n_neighbors: int = 10,
                 use_tags: bool = True,
                 use_authors: bool = True,
                 use_title: bool = True):
        """
        Initialize content-based recommender.
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features for TF-IDF
        min_df : int
            Minimum document frequency for TF-IDF
        max_df : float
            Maximum document frequency for TF-IDF
        n_neighbors : int
            Number of nearest neighbors for recommendations
        use_tags : bool
            Whether to include tags in content
        use_authors : bool
            Whether to include authors in content
        use_title : bool
            Whether to include title in content
        """
        super().__init__(name="ContentBasedRecommender")
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.n_neighbors = n_neighbors
        self.use_tags = use_tags
        self.use_authors = use_authors
        self.use_title = use_title
        
        # Model state
        self.books_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.knn_model = None
        self.item_mapping = None
        self.reverse_item_mapping = None
        self.item_col = None
    
    def fit(self, ratings_df: pd.DataFrame, books_df: Optional[pd.DataFrame] = None) -> 'ContentBasedRecommender':
        """
        Train the content-based recommender.
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings dataframe (used to get list of items)
        books_df : pd.DataFrame, optional
            Books metadata dataframe. If None, will try to load from data loader.
            
        Returns:
        --------
        self
        """
        # Determine item column
        self.item_col = 'book_id' if 'book_id' in ratings_df.columns else 'title'
        
        # Get unique items from ratings
        unique_items = ratings_df[self.item_col].unique()
        
        # Load books metadata if provided
        if books_df is None:
            # Try to load from data loader
            try:
                from ..data.data_loader import DataLoader
                loader = DataLoader()
                loader.load_books()
                loader.load_tags()
                books_df = loader.merge_books_with_tags()
            except Exception as e:
                warnings.warn(f"Could not load books metadata: {e}. Using ratings data only.")
                books_df = pd.DataFrame()
        
        if books_df is None or books_df.empty:
            # Fallback: create minimal books dataframe from ratings
            books_df = pd.DataFrame({
                self.item_col: unique_items
            })
            books_df['title'] = books_df[self.item_col]
            books_df['authors'] = ''
            books_df['tags'] = ''
        
        # Filter to only include items in ratings
        if self.item_col in books_df.columns:
            books_df = books_df[books_df[self.item_col].isin(unique_items)]
        elif 'title' in books_df.columns:
            books_df = books_df[books_df['title'].isin(unique_items)]
        
        self.books_df = books_df.copy()
        
        # Create content string
        self._create_content_features()
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.books_df['content'])
        
        # Fit KNN model
        self.knn_model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=min(self.n_neighbors + 1, len(self.books_df))
        )
        self.knn_model.fit(self.tfidf_matrix)
        
        # Create mappings
        self.item_mapping = {
            item: idx for idx, item in enumerate(self.books_df[self.item_col])
        }
        self.reverse_item_mapping = {
            idx: item for idx, item in enumerate(self.books_df[self.item_col])
        }
        
        return super().fit(ratings_df)
    
    def _create_content_features(self):
        """Create content string from book metadata."""
        # Initialize content column
        self.books_df['content'] = ''
        
        # Build content by concatenating selected features
        if self.use_title and 'title' in self.books_df.columns:
            self.books_df['content'] += ' ' + self.books_df['title'].fillna('').astype(str)
        
        if self.use_authors and 'authors' in self.books_df.columns:
            self.books_df['content'] += ' ' + self.books_df['authors'].fillna('').astype(str)
        
        if self.use_tags:
            if 'tags' in self.books_df.columns:
                self.books_df['content'] += ' ' + self.books_df['tags'].fillna('').astype(str)
            elif 'tag_name' in self.books_df.columns:
                self.books_df['content'] += ' ' + self.books_df['tag_name'].fillna('').astype(str)
        
        # Fallback: use item identifier if no content was added
        if self.books_df['content'].str.strip().eq('').all():
            item_col = self.item_col if self.item_col in self.books_df.columns else 'title'
            if item_col in self.books_df.columns:
                self.books_df['content'] = self.books_df[item_col].fillna('').astype(str)
        
        # Clean up: replace multiple spaces with single space
        self.books_df['content'] = self.books_df['content'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.
        For content-based, we return similarity-based score.
        
        Parameters:
        -----------
        user_id : str
            User identifier (not used in content-based, but required by interface)
        item_id : str
            Item identifier
            
        Returns:
        --------
        float
            Predicted rating (based on content similarity to user's rated items)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if item_id not in self.item_mapping:
            # Return default rating for unknown items
            return 3.0
        
        # For content-based, we need user's rating history to make meaningful predictions
        # This is a simplified version - in practice, you'd use user's rated items
        # For now, return average rating if available, else 3.0
        if self.books_df is not None and 'average_rating' in self.books_df.columns:
            item_idx = self.item_mapping[item_id]
            avg_rating = self.books_df.iloc[item_idx]['average_rating']
            if pd.notna(avg_rating):
                return float(avg_rating)
        
        return 3.0
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_rated: bool = True,
                 user_rated_items: Optional[List[str]] = None) -> List[Dict]:
        """
        Get top N recommendations for a user based on content similarity.
        
        Parameters:
        -----------
        user_id : str
            User identifier
        n_recommendations : int
            Number of recommendations to return
        exclude_rated : bool
            Whether to exclude items the user has already rated
        user_rated_items : List[str], optional
            List of items the user has rated (for similarity-based recommendations)
            
        Returns:
        --------
        List[Dict]
            List of recommendation dictionaries
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommendation")
        
        if user_rated_items is None or len(user_rated_items) == 0:
            # Cold-start: return popular items
            return self._get_popular_recommendations(n_recommendations)
        
        # Get user's rated items that exist in our database
        valid_rated_items = [item for item in user_rated_items if item in self.item_mapping]
        
        if len(valid_rated_items) == 0:
            return self._get_popular_recommendations(n_recommendations)
        
        # Compute average content vector for user's rated items
        rated_indices = [self.item_mapping[item] for item in valid_rated_items]
        user_profile_matrix = self.tfidf_matrix[rated_indices].mean(axis=0)
        # Convert to numpy array (handle both sparse and dense matrices)
        if hasattr(user_profile_matrix, 'toarray'):
            user_profile = user_profile_matrix.toarray().flatten()
        else:
            user_profile = np.asarray(user_profile_matrix).flatten()
        
        # Reshape for kneighbors (needs 2D array)
        user_profile = user_profile.reshape(1, -1)
        
        # Find similar items
        distances, indices = self.knn_model.kneighbors(
            user_profile,
            n_neighbors=min(n_recommendations * 2, len(self.books_df))
        )
        
        recommendations = []
        seen_items = set(valid_rated_items) if exclude_rated else set()
        
        for dist, idx in zip(distances[0], indices[0]):
            item_id = self.reverse_item_mapping[idx]
            
            if item_id in seen_items:
                continue
            
            seen_items.add(item_id)
            
            # Convert distance to similarity score (1 - distance)
            similarity = 1 - dist
            
            # Convert similarity to rating scale (1-5)
            predicted_rating = 1 + (similarity * 4)
            
            recommendations.append({
                'item_id': item_id,
                'predicted_rating': predicted_rating,
                'confidence': similarity,
                'similarity': similarity
            })
            
            if len(recommendations) >= n_recommendations:
                break
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations
    
    def get_similar_items(self, item_id: str, n_similar: int = 10) -> List[Dict]:
        """
        Get items similar to a given item.
        
        Parameters:
        -----------
        item_id : str
            Item identifier
        n_similar : int
            Number of similar items to return
            
        Returns:
        --------
        List[Dict]
            List of similar items with similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        if item_id not in self.item_mapping:
            return []
        
        item_idx = self.item_mapping[item_id]
        
        # Find similar items
        distances, indices = self.knn_model.kneighbors(
            self.tfidf_matrix[item_idx:item_idx+1],
            n_neighbors=min(n_similar + 1, len(self.books_df))
        )
        
        similar_items = []
        for dist, idx in zip(distances[0][1:], indices[0][1:]):  # Skip first (self)
            similar_item_id = self.reverse_item_mapping[idx]
            similarity = 1 - dist
            
            similar_items.append({
                'item_id': similar_item_id,
                'similarity': similarity,
                'predicted_rating': 1 + (similarity * 4)  # Convert to 1-5 scale
            })
        
        return similar_items
    
    def _get_popular_recommendations(self, n_recommendations: int) -> List[Dict]:
        """Get popular items as fallback."""
        if self.books_df is None or self.books_df.empty:
            return []
        
        # Sort by average rating if available
        if 'average_rating' in self.books_df.columns:
            sorted_books = self.books_df.sort_values('average_rating', ascending=False)
        else:
            # Just return first n items
            sorted_books = self.books_df.head(n_recommendations)
        
        recommendations = []
        for _, row in sorted_books.head(n_recommendations).iterrows():
            item_id = row[self.item_col]
            avg_rating = row.get('average_rating', 3.0)
            if pd.isna(avg_rating):
                avg_rating = 3.0
            
            recommendations.append({
                'item_id': item_id,
                'predicted_rating': float(avg_rating),
                'confidence': 0.3  # Low confidence for popularity-based
            })
        
        return recommendations

