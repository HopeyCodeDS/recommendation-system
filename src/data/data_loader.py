"""
Unified data loading and preprocessing module
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple, Dict
import warnings


class DataLoader:
    """
    Load and preprocess book recommendation data from multiple sources.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_dir : str
            Base directory containing data files
        """
        self.data_dir = data_dir
        self.ratings_df = None
        self.books_df = None
        self.tags_df = None
        self.book_tags_df = None
        self.processed_ratings_df = None
        self.processed_books_df = None
        
    def load_ratings(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load ratings data.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to ratings CSV file. If None, searches in common locations.
            
        Returns:
        --------
        pd.DataFrame
            Ratings dataframe with columns: user_id, book_id, rating
        """
        if filepath is None:
            # Try common locations
            possible_paths = [
                os.path.join(self.data_dir, "ratings.csv"),
                os.path.join(self.data_dir, "raw", "ratings.csv"),
                "submit/ratings.csv",
                "recommender_system/Books_rating.csv"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    filepath = path
                    break
            
            if filepath is None:
                raise FileNotFoundError("Ratings file not found in common locations")
        
        # Load ratings
        if "Books_rating.csv" in filepath:
            # Handle different format
            df = pd.read_csv(filepath)
            # Map columns to standard format
            if 'User_id' in df.columns and 'Title' in df.columns:
                # Need to map Title to book_id
                self.ratings_df = df[['User_id', 'Title', 'review/score']].copy()
                self.ratings_df.columns = ['user_id', 'title', 'rating']
            else:
                self.ratings_df = df
        else:
            self.ratings_df = pd.read_csv(filepath)
        
        # Standardize column names
        column_mapping = {
            'user_id': 'user_id',
            'User_id': 'user_id',
            'book_id': 'book_id',
            'Book_id': 'book_id',
            'Id': 'book_id',
            'rating': 'rating',
            'Rating': 'rating',
            'review/score': 'rating',
            'Title': 'title'
        }
        
        # Rename columns
        rename_dict = {}
        for col in self.ratings_df.columns:
            if col in column_mapping:
                rename_dict[col] = column_mapping[col]
        
        if rename_dict:
            self.ratings_df = self.ratings_df.rename(columns=rename_dict)
        
        # Ensure required columns exist
        if 'rating' not in self.ratings_df.columns:
            raise ValueError("Ratings file must contain a 'rating' column")
        
        return self.ratings_df
    
    def load_books(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load books metadata.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to books CSV file
            
        Returns:
        --------
        pd.DataFrame
            Books dataframe
        """
        if filepath is None:
            possible_paths = [
                os.path.join(self.data_dir, "books.csv"),
                os.path.join(self.data_dir, "raw", "books.csv"),
                "submit/books.csv",
                "data/books_data.csv"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    filepath = path
                    break
            
            if filepath is None:
                warnings.warn("Books file not found. Content-based filtering may be limited.")
                return pd.DataFrame()
        
        self.books_df = pd.read_csv(filepath)
        return self.books_df
    
    def load_tags(self, tags_filepath: Optional[str] = None,
                  book_tags_filepath: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load tags and book-tags mapping.
        
        Parameters:
        -----------
        tags_filepath : str, optional
            Path to tags CSV file
        book_tags_filepath : str, optional
            Path to book_tags CSV file
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tags dataframe and book_tags dataframe
        """
        # Load tags
        if tags_filepath is None:
            possible_paths = [
                os.path.join(self.data_dir, "tags.csv"),
                os.path.join(self.data_dir, "raw", "tags.csv"),
                "submit/tags.csv"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    tags_filepath = path
                    break
        
        if tags_filepath and os.path.exists(tags_filepath):
            self.tags_df = pd.read_csv(tags_filepath)
        else:
            warnings.warn("Tags file not found.")
            self.tags_df = pd.DataFrame()
        
        # Load book_tags
        if book_tags_filepath is None:
            possible_paths = [
                os.path.join(self.data_dir, "book_tags.csv"),
                os.path.join(self.data_dir, "raw", "book_tags.csv"),
                "submit/book_tags.csv"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    book_tags_filepath = path
                    break
        
        if book_tags_filepath and os.path.exists(book_tags_filepath):
            self.book_tags_df = pd.read_csv(book_tags_filepath)
        else:
            warnings.warn("Book tags file not found.")
            self.book_tags_df = pd.DataFrame()
        
        return self.tags_df, self.book_tags_df
    
    def preprocess_ratings(self, min_user_ratings: int = 3, 
                          min_book_ratings: int = 3,
                          rating_range: Tuple[float, float] = (1.0, 5.0)) -> pd.DataFrame:
        """
        Preprocess ratings data: filter, clean, and validate.
        
        Parameters:
        -----------
        min_user_ratings : int
            Minimum number of ratings per user
        min_book_ratings : int
            Minimum number of ratings per book
        rating_range : Tuple[float, float]
            Valid rating range (min, max)
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed ratings dataframe
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data must be loaded first")
        
        df = self.ratings_df.copy()
        
        # Remove rows with missing values
        initial_count = len(df)
        df = df.dropna(subset=['rating'])
        
        # Filter by rating range
        df = df[(df['rating'] >= rating_range[0]) & 
                (df['rating'] <= rating_range[1])]
        
        # If we have book_id, filter by minimum ratings
        if 'book_id' in df.columns:
            # Filter books with minimum ratings
            book_counts = df['book_id'].value_counts()
            valid_books = book_counts[book_counts >= min_book_ratings].index
            df = df[df['book_id'].isin(valid_books)]
            
            # Filter users with minimum ratings
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_user_ratings].index
            df = df[df['user_id'].isin(valid_users)]
        elif 'title' in df.columns:
            # Filter books (by title) with minimum ratings
            book_counts = df['title'].value_counts()
            valid_books = book_counts[book_counts >= min_book_ratings].index
            df = df[df['title'].isin(valid_books)]
            
            # Filter users with minimum ratings
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_user_ratings].index
            df = df[df['user_id'].isin(valid_users)]
        
        self.processed_ratings_df = df.reset_index(drop=True)
        
        print(f"Preprocessing: {initial_count} -> {len(self.processed_ratings_df)} ratings "
              f"({len(self.processed_ratings_df) / initial_count * 100:.1f}% retained)")
        
        return self.processed_ratings_df
    
    def merge_books_with_tags(self) -> pd.DataFrame:
        """
        Merge books metadata with tags.
        
        Returns:
        --------
        pd.DataFrame
            Books dataframe with aggregated tags
        """
        if self.books_df is None or self.books_df.empty:
            return pd.DataFrame()
        
        if self.book_tags_df is None or self.book_tags_df.empty:
            return self.books_df
        
        if self.tags_df is None or self.tags_df.empty:
            return self.books_df
        
        # Merge book_tags with tags to get tag names
        book_tags_merged = self.book_tags_df.merge(
            self.tags_df,
            on='tag_id',
            how='left'
        )
        
        # Aggregate tags per book
        # Use goodreads_book_id or book_id
        book_id_col = 'goodreads_book_id' if 'goodreads_book_id' in self.book_tags_df.columns else 'book_id'
        
        book_tags_agg = book_tags_merged.groupby(book_id_col)['tag_name'].apply(
            lambda x: ' '.join(x.astype(str).dropna())
        ).reset_index()
        book_tags_agg.columns = [book_id_col, 'tags']
        
        # Merge with books dataframe
        merge_col = 'goodreads_book_id' if 'goodreads_book_id' in self.books_df.columns else 'book_id'
        
        if merge_col in self.books_df.columns:
            self.processed_books_df = self.books_df.merge(
                book_tags_agg,
                left_on=merge_col,
                right_on=book_id_col,
                how='left'
            )
            self.processed_books_df['tags'] = self.processed_books_df['tags'].fillna('')
        else:
            self.processed_books_df = self.books_df.copy()
            self.processed_books_df['tags'] = ''
        
        return self.processed_books_df
    
    def get_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get all processed dataframes.
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing processed dataframes
        """
        return {
            'ratings': self.processed_ratings_df if self.processed_ratings_df is not None else self.ratings_df,
            'books': self.processed_books_df if self.processed_books_df is not None else self.books_df,
            'tags': self.tags_df,
            'book_tags': self.book_tags_df
        }
    
    def save_processed_data(self, output_dir: str = "data/processed"):
        """
        Save processed data to disk.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for processed data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        data = self.get_processed_data()
        
        for name, df in data.items():
            if df is not None and not df.empty:
                filepath = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(filepath, index=False)
                print(f"Saved {name} to {filepath}")

