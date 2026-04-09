"""
Recommendations endpoints.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException

from ..model_loader import ModelStore, get_store
from ..schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
    SimilarItemResponse,
)

router = APIRouter()

# Human-readable explanations per algorithm
_EXPLANATIONS = {
    "cf-user": "Based on ratings from similar users",
    "cf-item": "Because users who liked similar books also liked this",
    "content": "Matches your reading profile based on title, author & genre tags",
    "hybrid": "Blended from collaborative and content signals",
}


def _explanation(algorithm: str, confidence: float) -> str:
    base = _EXPLANATIONS.get(algorithm, "Recommended for you")
    level = "strong" if confidence >= 0.7 else "moderate" if confidence >= 0.4 else "weak"
    return f"{base} ({level} match)"


def _pick_model(algorithm: str, store: ModelStore, cf_weight: float | None = None):
    """Return the appropriate recommender, optionally patching hybrid weight."""
    if algorithm == "cf-user":
        return store.cf_uu
    if algorithm == "cf-item":
        return store.cf_ii
    if algorithm == "content":
        return store.content_based
    if algorithm == "hybrid":
        if cf_weight is not None:
            # Temporarily override weights without mutating the stored model
            import copy
            h = copy.copy(store.hybrid)
            cb_weight = 1.0 - cf_weight
            h.cf_weight = cf_weight
            h.cb_weight = cb_weight
            return h
        return store.hybrid
    raise HTTPException(status_code=400, detail=f"Unknown algorithm '{algorithm}'. Use: cf-user, cf-item, content, hybrid")


@router.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(req: RecommendationRequest, store: ModelStore = Depends(get_store)):
    model = _pick_model(req.algorithm, store, req.cf_weight)

    # For content-based, we need to pass the user's rated items explicitly
    user_id = req.user_id
    kwargs = {"n_recommendations": req.n, "exclude_rated": True}

    if req.algorithm == "content":
        rated = store.content_based.ratings_df
        if rated is not None:
            user_rows = rated[rated["user_id"] == user_id]
            item_col = store.content_based.item_col or "book_id"
            user_rated_items = user_rows[item_col].tolist() if not user_rows.empty else []
        else:
            user_rated_items = []
        kwargs["user_rated_items"] = user_rated_items

    try:
        raw = model.recommend(user_id, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    items = []
    for rank, rec in enumerate(raw, start=1):
        bid = int(rec["item_id"]) if hasattr(rec["item_id"], "item") else rec["item_id"]
        book = store.get_book(bid)
        confidence = float(rec.get("confidence", rec.get("similarity", 0.5)))
        items.append(
            RecommendationItem(
                rank=rank,
                item_id=bid,
                title=book.get("title") if book else str(bid),
                authors=book.get("authors") if book else None,
                image_url=book.get("image_url") if book else None,
                average_rating=book.get("average_rating") if book else None,
                predicted_rating=round(float(rec["predicted_rating"]), 2),
                confidence=round(confidence, 4),
                explanation=_explanation(req.algorithm, confidence),
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        algorithm=req.algorithm,
        recommendations=items,
    )


@router.get("/recommendations/similar/{book_id}", response_model=List[SimilarItemResponse])
def similar_books(book_id: int, n: int = 8, store: ModelStore = Depends(get_store)):
    book = store.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    raw = store.content_based.get_similar_items(book_id, n_similar=n)
    result = []
    for s in raw:
        b = store.get_book(s["item_id"])
        result.append(
            SimilarItemResponse(
                item_id=s["item_id"],
                title=b.get("title") if b else None,
                authors=b.get("authors") if b else None,
                image_url=b.get("image_url") if b else None,
                similarity_score=round(s["similarity"], 4),
            )
        )
    return result


@router.get("/recommendations/cold-start", response_model=RecommendationResponse)
def cold_start(n: int = 10, store: ModelStore = Depends(get_store)):
    """Return popularity-based recommendations for users with no history."""
    raw = store.content_based._get_popular_recommendations(n)
    items = []
    for rank, rec in enumerate(raw, start=1):
        bid = int(rec["item_id"]) if hasattr(rec["item_id"], "item") else rec["item_id"]
        book = store.get_book(bid)
        items.append(
            RecommendationItem(
                rank=rank,
                item_id=bid,
                title=book.get("title") if book else str(bid),
                authors=book.get("authors") if book else None,
                image_url=book.get("image_url") if book else None,
                average_rating=book.get("average_rating") if book else None,
                predicted_rating=round(float(rec["predicted_rating"]), 2),
                confidence=float(rec.get("confidence", 0.3)),
                explanation="Popular book — highly rated by many readers",
            )
        )
    return RecommendationResponse(user_id="cold-start", algorithm="popular", recommendations=items)
