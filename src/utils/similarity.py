"""
Similarity computation utilities
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union


def cosine_similarity_sparse(matrix: Union[np.ndarray, csr_matrix], 
                            dense_output: bool = False) -> Union[np.ndarray, csr_matrix]:
    """
    Compute cosine similarity for sparse or dense matrices.
    
    Parameters:
    -----------
    matrix : np.ndarray or csr_matrix
        Input matrix (users x items or items x items)
    dense_output : bool
        Whether to return dense output (default: False for sparse input)
        
    Returns:
    --------
    np.ndarray or csr_matrix
        Similarity matrix
    """
    if issparse(matrix):
        # Check if matrix is empty
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Cannot compute similarity for empty matrix")
        
        # For sparse matrices, convert to dense for sklearn's cosine_similarity
        # This is memory-intensive but necessary for accurate similarity
        dense_matrix = matrix.toarray()
        
        # Check if dense matrix is empty or has no valid data
        if dense_matrix.size == 0:
            raise ValueError("Cannot compute similarity for empty matrix")
        
        similarity = cosine_similarity(dense_matrix, dense_matrix)
        if not dense_output:
            return csr_matrix(similarity)
        return similarity
    else:
        if matrix.size == 0 or matrix.shape[0] == 0:
            raise ValueError("Cannot compute similarity for empty matrix")
        return cosine_similarity(matrix, matrix)


def pearson_correlation(matrix: np.ndarray, 
                       min_common_items: int = 1) -> np.ndarray:
    """
    Compute Pearson correlation similarity matrix.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix (users x items)
    min_common_items : int
        Minimum number of common items required for similarity computation
        
    Returns:
    --------
    np.ndarray
        Correlation similarity matrix
    """
    # Mean-center each row (user)
    mean_centered = matrix - matrix.mean(axis=1, keepdims=True)
    
    # Compute correlation
    std = np.std(mean_centered, axis=1, keepdims=True)
    std[std == 0] = 1  # Avoid division by zero
    
    normalized = mean_centered / std
    
    # Compute correlation matrix
    correlation = np.dot(normalized, normalized.T) / matrix.shape[1]
    
    # Set correlations to 0 where there are too few common items
    # This is a simplified check - in practice, you'd check actual common items
    common_items = np.dot((matrix != 0).astype(int), (matrix != 0).T.astype(int))
    correlation[common_items < min_common_items] = 0
    
    return correlation


def apply_similarity_threshold(similarity_matrix: np.ndarray, 
                              threshold: float = 0.0) -> np.ndarray:
    """
    Apply threshold to similarity matrix, setting values below threshold to 0.
    
    Parameters:
    -----------
    similarity_matrix : np.ndarray
        Similarity matrix
    threshold : float
        Minimum similarity threshold
        
    Returns:
    --------
    np.ndarray
        Thresholded similarity matrix
    """
    thresholded = similarity_matrix.copy()
    thresholded[thresholded < threshold] = 0
    return thresholded


def normalize_similarity_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize similarity matrix by row sums.
    
    Parameters:
    -----------
    similarity_matrix : np.ndarray
        Similarity matrix
        
    Returns:
    --------
    np.ndarray
        Normalized similarity matrix
    """
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return similarity_matrix / row_sums

