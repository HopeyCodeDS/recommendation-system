"""
Data quality validation and statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy.sparse import csr_matrix


class DataValidator:
    """
    Validate data quality and compute statistics for recommender systems.
    """
    
    def __init__(self, ratings_df: pd.DataFrame):
        """
        Initialize validator with ratings data.
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings dataframe with columns: user_id, book_id (or title), rating
        """
        self.ratings_df = ratings_df.copy()
        self.stats = {}
        
    def validate(self) -> Dict:
        """
        Run all validation checks and compute statistics.
        
        Returns:
        --------
        Dict
            Dictionary containing validation results and statistics
        """
        self.stats = {
            'basic_stats': self.compute_basic_stats(),
            'sparsity': self.compute_sparsity(),
            'rating_distribution': self.compute_rating_distribution(),
            'user_stats': self.compute_user_stats(),
            'item_stats': self.compute_item_stats(),
            'warnings': []
        }
        
        # Check for common issues
        self._check_data_quality()
        
        return self.stats
    
    def compute_basic_stats(self) -> Dict:
        """Compute basic statistics."""
        stats = {
            'total_ratings': len(self.ratings_df),
            'unique_users': self.ratings_df['user_id'].nunique(),
            'unique_items': self._get_item_column().nunique(),
            'avg_rating': self.ratings_df['rating'].mean(),
            'std_rating': self.ratings_df['rating'].std(),
            'min_rating': self.ratings_df['rating'].min(),
            'max_rating': self.ratings_df['rating'].max()
        }
        
        stats['avg_ratings_per_user'] = stats['total_ratings'] / stats['unique_users']
        stats['avg_ratings_per_item'] = stats['total_ratings'] / stats['unique_items']
        
        return stats
    
    def compute_sparsity(self) -> Dict:
        """Compute sparsity metrics."""
        item_col = self._get_item_column()
        
        # Create user-item matrix
        user_item_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns=item_col.name if hasattr(item_col, 'name') else 'item',
            values='rating',
            fill_value=0
        )
        
        total_cells = user_item_matrix.shape[0] * user_item_matrix.shape[1]
        non_zero_cells = (user_item_matrix != 0).sum().sum()
        
        sparsity = 1 - (non_zero_cells / total_cells)
        
        return {
            'sparsity': sparsity,
            'density': 1 - sparsity,
            'matrix_shape': user_item_matrix.shape,
            'total_cells': total_cells,
            'non_zero_cells': non_zero_cells
        }
    
    def compute_rating_distribution(self) -> Dict:
        """Compute rating distribution statistics."""
        rating_counts = self.ratings_df['rating'].value_counts().sort_index()
        
        return {
            'distribution': rating_counts.to_dict(),
            'most_common_rating': rating_counts.idxmax(),
            'rating_entropy': self._compute_entropy(rating_counts)
        }
    
    def compute_user_stats(self) -> Dict:
        """Compute user-related statistics."""
        user_stats = self.ratings_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std']
        })
        user_stats.columns = ['num_ratings', 'avg_rating', 'std_rating']
        
        return {
            'ratings_per_user': {
                'mean': user_stats['num_ratings'].mean(),
                'median': user_stats['num_ratings'].median(),
                'std': user_stats['num_ratings'].std(),
                'min': user_stats['num_ratings'].min(),
                'max': user_stats['num_ratings'].max()
            },
            'avg_rating_per_user': {
                'mean': user_stats['avg_rating'].mean(),
                'std': user_stats['avg_rating'].std()
            }
        }
    
    def compute_item_stats(self) -> Dict:
        """Compute item-related statistics."""
        item_col = self._get_item_column()
        item_stats = self.ratings_df.groupby(item_col).agg({
            'rating': ['count', 'mean', 'std']
        })
        item_stats.columns = ['num_ratings', 'avg_rating', 'std_rating']
        
        return {
            'ratings_per_item': {
                'mean': item_stats['num_ratings'].mean(),
                'median': item_stats['num_ratings'].median(),
                'std': item_stats['num_ratings'].std(),
                'min': item_stats['num_ratings'].min(),
                'max': item_stats['num_ratings'].max()
            },
            'avg_rating_per_item': {
                'mean': item_stats['avg_rating'].mean(),
                'std': item_stats['avg_rating'].std()
            }
        }
    
    def _get_item_column(self) -> pd.Series:
        """Get the item column (book_id or title)."""
        if 'book_id' in self.ratings_df.columns:
            return self.ratings_df['book_id']
        elif 'title' in self.ratings_df.columns:
            return self.ratings_df['title']
        else:
            raise ValueError("No item column found (book_id or title)")
    
    def _compute_entropy(self, counts: pd.Series) -> float:
        """Compute entropy of rating distribution."""
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _check_data_quality(self):
        """Check for common data quality issues."""
        warnings = []
        
        # Check for high sparsity
        if self.stats['sparsity']['sparsity'] > 0.99:
            warnings.append("Very high sparsity (>99%). Recommendations may be limited.")
        
        # Check for rating imbalance
        rating_dist = self.stats['rating_distribution']['distribution']
        if len(rating_dist) > 0:
            max_rating_count = max(rating_dist.values())
            total_ratings = sum(rating_dist.values())
            if max_rating_count / total_ratings > 0.5:
                warnings.append("Rating distribution is highly imbalanced.")
        
        # Check for users with very few ratings
        user_stats = self.stats['user_stats']['ratings_per_user']
        if user_stats['median'] < 3:
            warnings.append("Many users have very few ratings. Consider filtering.")
        
        # Check for items with very few ratings
        item_stats = self.stats['item_stats']['ratings_per_item']
        if item_stats['median'] < 3:
            warnings.append("Many items have very few ratings. Consider filtering.")
        
        self.stats['warnings'] = warnings
    
    def print_report(self):
        """Print a formatted validation report."""
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        
        print("\nBasic Statistics:")
        print(f"  Total Ratings: {self.stats['basic_stats']['total_ratings']:,}")
        print(f"  Unique Users: {self.stats['basic_stats']['unique_users']:,}")
        print(f"  Unique Items: {self.stats['basic_stats']['unique_items']:,}")
        print(f"  Average Rating: {self.stats['basic_stats']['avg_rating']:.2f}")
        print(f"  Avg Ratings per User: {self.stats['basic_stats']['avg_ratings_per_user']:.2f}")
        print(f"  Avg Ratings per Item: {self.stats['basic_stats']['avg_ratings_per_item']:.2f}")
        
        print("\nSparsity:")
        print(f"  Sparsity: {self.stats['sparsity']['sparsity']:.4f} ({self.stats['sparsity']['sparsity']*100:.2f}%)")
        print(f"  Density: {self.stats['sparsity']['density']:.4f} ({self.stats['sparsity']['density']*100:.2f}%)")
        print(f"  Matrix Shape: {self.stats['sparsity']['matrix_shape']}")
        
        print("\nRating Distribution:")
        for rating, count in sorted(self.stats['rating_distribution']['distribution'].items()):
            print(f"  Rating {rating}: {count:,} ({count/self.stats['basic_stats']['total_ratings']*100:.1f}%)")
        
        if self.stats['warnings']:
            print("\nWarnings:")
            for warning in self.stats['warnings']:
                print(f"  ⚠ {warning}")
        
        print("=" * 60)

