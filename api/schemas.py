"""
Pydantic schemas for request/response models.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ── Books ──────────────────────────────────────────────────────────────────────

class BookResponse(BaseModel):
    book_id: int
    title: str
    authors: Optional[str] = None
    average_rating: Optional[float] = None
    image_url: Optional[str] = None
    small_image_url: Optional[str] = None
    original_publication_year: Optional[float] = None
    language_code: Optional[str] = None
    tags: Optional[str] = None


class BookDetailResponse(BookResponse):
    similar_books: List["SimilarBookResponse"] = []


class SimilarBookResponse(BaseModel):
    book_id: Any
    title: Optional[str] = None
    authors: Optional[str] = None
    image_url: Optional[str] = None
    similarity_score: float


# ── Users ──────────────────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    user_id: Any
    display_name: str
    rating_count: int
    avg_rating: float
    profile_type: str


# ── Recommendations ────────────────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    user_id: Any
    algorithm: str = Field(
        default="hybrid",
        description="One of: cf-user, cf-item, content, hybrid",
    )
    n: int = Field(default=10, ge=1, le=20)
    cf_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom CF weight for hybrid (0-1). If omitted, uses adaptive weighting.",
    )


class RecommendationItem(BaseModel):
    rank: int
    item_id: Any
    title: Optional[str] = None
    authors: Optional[str] = None
    image_url: Optional[str] = None
    average_rating: Optional[float] = None
    predicted_rating: float
    confidence: Optional[float] = None
    explanation: str


class RecommendationResponse(BaseModel):
    user_id: Any
    algorithm: str
    recommendations: List[RecommendationItem]


# ── Metrics ────────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]


# ── Similarity ─────────────────────────────────────────────────────────────────

class SimilarityMatrixResponse(BaseModel):
    users: List[str]
    matrix: List[List[float]]


class SimilarItemResponse(BaseModel):
    item_id: Any
    title: Optional[str] = None
    authors: Optional[str] = None
    image_url: Optional[str] = None
    similarity_score: float


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    dataset_stats: Dict[str, int]
