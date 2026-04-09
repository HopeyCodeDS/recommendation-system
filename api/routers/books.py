"""
Books and Users endpoints.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException

from ..model_loader import ModelStore, get_store
from ..schemas import BookResponse, BookDetailResponse, SimilarBookResponse, UserResponse

router = APIRouter()


def _book_to_response(b: dict) -> BookResponse:
    return BookResponse(
        book_id=b.get("book_id"),
        title=b.get("title") or "Unknown",
        authors=b.get("authors"),
        average_rating=b.get("average_rating"),
        image_url=b.get("image_url"),
        small_image_url=b.get("small_image_url"),
        original_publication_year=b.get("original_publication_year"),
        language_code=b.get("language_code"),
        tags=b.get("tags"),
    )


@router.get("/books", response_model=List[BookResponse])
def list_books(store: ModelStore = Depends(get_store)):
    return [_book_to_response(b) for b in store.books_catalog]


@router.get("/books/{book_id}", response_model=BookDetailResponse)
def get_book(book_id: int, store: ModelStore = Depends(get_store)):
    book = store.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    # Get similar books from content-based model
    similar_raw = store.content_based.get_similar_items(book_id, n_similar=6)
    similar = []
    for s in similar_raw:
        b = store.get_book(s["item_id"])
        similar.append(
            SimilarBookResponse(
                book_id=s["item_id"],
                title=b.get("title") if b else None,
                authors=b.get("authors") if b else None,
                image_url=b.get("image_url") if b else None,
                similarity_score=round(s["similarity"], 4),
            )
        )

    base = _book_to_response(book)
    return BookDetailResponse(**base.model_dump(), similar_books=similar)


@router.get("/users", response_model=List[UserResponse])
def list_users(store: ModelStore = Depends(get_store)):
    return [UserResponse(**u) for u in store.users_catalog]
