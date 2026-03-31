"""
Evaluation metrics for recommender systems
"""

import numpy as np
from typing import List, Set, Dict


def precision_at_k(recommended_items: List[str],
                  relevant_items: Set[str],
                  k: int) -> float:
    """
    Compute Precision@K.
    
    Parameters:
    -----------
    recommended_items : List[str]
        List of recommended item IDs
    relevant_items : Set[str]
        Set of relevant (actually liked) item IDs
    k : int
        Number of top recommendations to consider
        
    Returns:
    --------
    float
        Precision@K score
    """
    if k == 0 or len(recommended_items) == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    relevant_recommended = len([item for item in top_k if item in relevant_items])
    
    return relevant_recommended / min(k, len(recommended_items))


def recall_at_k(recommended_items: List[str],
               relevant_items: Set[str],
               k: int) -> float:
    """
    Compute Recall@K.
    
    Parameters:
    -----------
    recommended_items : List[str]
        List of recommended item IDs
    relevant_items : Set[str]
        Set of relevant (actually liked) item IDs
    k : int
        Number of top recommendations to consider
        
    Returns:
    --------
    float
        Recall@K score
    """
    if len(relevant_items) == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    relevant_recommended = len([item for item in top_k if item in relevant_items])
    
    return relevant_recommended / len(relevant_items)


def f1_at_k(recommended_items: List[str],
            relevant_items: Set[str],
            k: int) -> float:
    """
    Compute F1@K.
    
    Parameters:
    -----------
    recommended_items : List[str]
        List of recommended item IDs
    relevant_items : Set[str]
        Set of relevant (actually liked) item IDs
    k : int
        Number of top recommendations to consider
        
    Returns:
    --------
    float
        F1@K score
    """
    precision = precision_at_k(recommended_items, relevant_items, k)
    recall = recall_at_k(recommended_items, relevant_items, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def rmse(predictions: List[float], actuals: List[float]) -> float:
    """
    Compute Root Mean Squared Error.
    
    Parameters:
    -----------
    predictions : List[float]
        Predicted ratings
    actuals : List[float]
        Actual ratings
        
    Returns:
    --------
    float
        RMSE score
    """
    if len(predictions) == 0:
        return np.nan
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return np.sqrt(np.mean((predictions - actuals) ** 2))


def mae(predictions: List[float], actuals: List[float]) -> float:
    """
    Compute Mean Absolute Error.
    
    Parameters:
    -----------
    predictions : List[float]
        Predicted ratings
    actuals : List[float]
        Actual ratings
        
    Returns:
    --------
    float
        MAE score
    """
    if len(predictions) == 0:
        return np.nan
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return np.mean(np.abs(predictions - actuals))


def coverage(recommended_items_all_users: List[List[str]],
            all_items: Set[str]) -> float:
    """
    Compute Coverage: fraction of items that can be recommended.
    
    Parameters:
    -----------
    recommended_items_all_users : List[List[str]]
        List of recommendations for each user
    all_items : Set[str]
        Set of all items in the catalog
        
    Returns:
    --------
    float
        Coverage score (0-1)
    """
    if len(all_items) == 0:
        return 0.0
    
    recommended_items_set = set()
    for user_recs in recommended_items_all_users:
        recommended_items_set.update(user_recs)
    
    return len(recommended_items_set) / len(all_items)


def diversity(recommended_items: List[str],
             item_similarity_matrix: np.ndarray = None,
             item_mapping: Dict[str, int] = None) -> float:
    """
    Compute Diversity: average pairwise dissimilarity of recommendations.
    
    Parameters:
    -----------
    recommended_items : List[str]
        List of recommended item IDs
    item_similarity_matrix : np.ndarray, optional
        Item-item similarity matrix
    item_mapping : Dict[str, int], optional
        Mapping from item ID to matrix index
        
    Returns:
    --------
    float
        Diversity score (higher = more diverse)
    """
    if len(recommended_items) < 2:
        return 0.0
    
    if item_similarity_matrix is None or item_mapping is None:
        # Simple diversity: count unique items (normalized)
        unique_items = len(set(recommended_items))
        return unique_items / len(recommended_items)
    
    # Compute average pairwise dissimilarity
    similarities = []
    for i, item1 in enumerate(recommended_items):
        if item1 not in item_mapping:
            continue
        idx1 = item_mapping[item1]
        for item2 in recommended_items[i+1:]:
            if item2 not in item_mapping:
                continue
            idx2 = item_mapping[item2]
            similarity = item_similarity_matrix[idx1, idx2]
            similarities.append(1 - similarity)  # Dissimilarity
    
    if len(similarities) == 0:
        return 0.0
    
    return np.mean(similarities)


def novelty(recommended_items: List[str],
           item_popularity: Dict[str, float]) -> float:
    """
    Compute Novelty: average negative log popularity of recommendations.
    
    Parameters:
    -----------
    recommended_items : List[str]
        List of recommended item IDs
    item_popularity : Dict[str, float]
        Dictionary mapping item ID to popularity (0-1)
        
    Returns:
    --------
    float
        Novelty score (higher = more novel/less popular)
    """
    if len(recommended_items) == 0:
        return 0.0
    
    novelty_scores = []
    for item in recommended_items:
        popularity = item_popularity.get(item, 0.5)
        # Avoid log(0)
        popularity = max(popularity, 1e-10)
        novelty_scores.append(-np.log2(popularity))
    
    return np.mean(novelty_scores)


def mean_reciprocal_rank(recommended_items: List[str],
                        relevant_items: Set[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Parameters:
    -----------
    recommended_items : List[str]
        List of recommended item IDs
    relevant_items : Set[str]
        Set of relevant item IDs
        
    Returns:
    --------
    float
        MRR score
    """
    for rank, item in enumerate(recommended_items, 1):
        if item in relevant_items:
            return 1.0 / rank
    
    return 0.0


def ndcg_at_k(recommended_items: List[str],
              relevant_items: Set[str],
              k: int,
              item_scores: Dict[str, float] = None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain@K.
    
    Parameters:
    -----------
    recommended_items : List[str]
        List of recommended item IDs
    relevant_items : Set[str]
        Set of relevant item IDs
    k : int
        Number of top recommendations to consider
    item_scores : Dict[str, float], optional
        Relevance scores for items (default: binary)
        
    Returns:
    --------
    float
        NDCG@K score
    """
    if k == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    
    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(top_k, 1):
        if item in relevant_items:
            if item_scores:
                relevance = item_scores.get(item, 1.0)
            else:
                relevance = 1.0
            dcg += relevance / np.log2(i + 1)
    
    # Compute IDCG (ideal DCG): assume best-case ranking of ALL relevant items, up to k positions
    if item_scores:
        ideal_relevance = sorted(
            [item_scores.get(item, 1.0) for item in relevant_items],
            reverse=True
        )[:k]
    else:
        ideal_relevance = [1.0] * min(len(relevant_items), k)

    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

