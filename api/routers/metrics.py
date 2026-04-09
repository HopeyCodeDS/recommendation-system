"""
Metrics endpoints — serve precomputed comparison and similarity data.
"""

from fastapi import APIRouter, Depends
from ..model_loader import ModelStore, get_store
from ..schemas import MetricsResponse, SimilarityMatrixResponse

router = APIRouter()


@router.get("/metrics/comparison", response_model=MetricsResponse)
def metrics_comparison(store: ModelStore = Depends(get_store)):
    """Return precomputed evaluation metrics for all algorithms."""
    return MetricsResponse(metrics=store.precomputed_metrics)


@router.get("/similarity/users", response_model=SimilarityMatrixResponse)
def user_similarity_matrix(store: ModelStore = Depends(get_store)):
    """Return the user-user cosine similarity matrix from the trained CF model."""
    return SimilarityMatrixResponse(
        users=store.similarity_matrices.get("users", []),
        matrix=store.similarity_matrices.get("matrix", []),
    )


@router.get("/tfidf/{book_id}")
def tfidf_terms(book_id: str, store: ModelStore = Depends(get_store)):
    """Return top TF-IDF terms for a given book (for the How It Works visualizer)."""
    terms = store.tfidf_top_terms.get(book_id) or store.tfidf_top_terms.get(str(book_id)) or []
    return {"book_id": book_id, "terms": terms}
