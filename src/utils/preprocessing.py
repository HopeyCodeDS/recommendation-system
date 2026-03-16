"""
Data preprocessing utilities
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Dict


def normalize_ratings(ratings_df: pd.DataFrame, 
                     method: str = 'mean_centering') -> pd.DataFrame:
    """
    Normalize ratings by user.
    
    Parameters:
    -----------
    ratings_df : pd.DataFrame
        Ratings dataframe with columns: user_id, item_id (or title), rating
    method : str
        Normalization method: 'mean_centering', 'z_score', or 'none'
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with normalized ratings
    """
    df = ratings_df.copy()
    
    if method == 'mean_centering':
        user_means = df.groupby('user_id')['rating'].transform('mean')
        df['normalized_rating'] = df['rating'] - user_means
    elif method == 'z_score':
        user_means = df.groupby('user_id')['rating'].transform('mean')
        user_stds = df.groupby('user_id')['rating'].transform('std')
        user_stds[user_stds == 0] = 1  # Avoid division by zero
        df['normalized_rating'] = (df['rating'] - user_means) / user_stds
    else:
        df['normalized_rating'] = df['rating']
    
    return df


def filter_sparse_data(ratings_df: pd.DataFrame,
                      min_user_ratings: int = 3,
                      min_item_ratings: int = 3) -> pd.DataFrame:
    """
    Filter users and items with too few ratings.
    
    Parameters:
    -----------
    ratings_df : pd.DataFrame
        Ratings dataframe
    min_user_ratings : int
        Minimum number of ratings per user
    min_item_ratings : int
        Minimum number of ratings per item
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    df = ratings_df.copy()
    
    # Determine item column
    item_col = 'book_id' if 'book_id' in df.columns else 'title'
    
    # Filter items
    item_counts = df[item_col].value_counts()
    valid_items = item_counts[item_counts >= min_item_ratings].index
    df = df[df[item_col].isin(valid_items)]
    
    # Filter users
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df['user_id'].isin(valid_users)]
    
    return df.reset_index(drop=True)


def create_user_item_matrix(ratings_df: pd.DataFrame,
                           item_col: str = None) -> Tuple[csr_matrix, Dict, Dict]:
    """
    Create sparse user-item matrix from ratings dataframe.
    
    Parameters:
    -----------
    ratings_df : pd.DataFrame
        Ratings dataframe with user_id, item column, and rating
    item_col : str, optional
        Name of item column. If None, auto-detect (book_id or title)
        
    Returns:
    --------
    Tuple[csr_matrix, Dict, Dict]
        Sparse matrix, user mapping, item mapping
    """
    if item_col is None:
        item_col = 'book_id' if 'book_id' in ratings_df.columns else 'title'
    
    # Create mappings
    unique_users = ratings_df['user_id'].unique()
    unique_items = ratings_df[item_col].unique()
    
    user_mapping = {user: idx for idx, user in enumerate(unique_users)}
    item_mapping = {item: idx for idx, item in enumerate(unique_items)}
    
    # Create matrix indices
    rows = [user_mapping[user] for user in ratings_df['user_id']]
    cols = [item_mapping[item] for item in ratings_df[item_col]]
    
    # Get rating values
    if 'normalized_rating' in ratings_df.columns:
        values = ratings_df['normalized_rating'].values
    else:
        values = ratings_df['rating'].values
    
    # Create sparse matrix
    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_mapping), len(item_mapping))
    )
    
    return matrix, user_mapping, item_mapping

