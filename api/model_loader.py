"""
Model loader — loads serialized joblib models and JSON data files at startup.
All routers access state via get_models() dependency.
"""

import os
import json
import joblib
from typing import Any, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


class ModelStore:
    """Central store for all pre-trained models and precomputed data."""

    def __init__(self):
        self.cf_uu = None
        self.cf_ii = None
        self.content_based = None
        self.hybrid = None

        self.books_catalog: list = []
        self.users_catalog: list = []
        self.precomputed_metrics: dict = {}
        self.similarity_matrices: dict = {}
        self.tfidf_top_terms: dict = {}

        # Quick-lookup dicts
        self._books_by_id: Dict[Any, dict] = {}

    def load(self):
        """Load all models and data files."""
        # ── Models ────────────────────────────────────────────────────────────
        self.cf_uu = joblib.load(os.path.join(MODELS_DIR, "cf_user_user.joblib"))
        self.cf_ii = joblib.load(os.path.join(MODELS_DIR, "cf_item_item.joblib"))
        self.content_based = joblib.load(os.path.join(MODELS_DIR, "content_based.joblib"))
        self.hybrid = joblib.load(os.path.join(MODELS_DIR, "hybrid.joblib"))

        # ── Data files ────────────────────────────────────────────────────────
        with open(os.path.join(DATA_DIR, "books_catalog.json")) as f:
            self.books_catalog = json.load(f)

        with open(os.path.join(DATA_DIR, "users_catalog.json")) as f:
            self.users_catalog = json.load(f)

        with open(os.path.join(DATA_DIR, "precomputed_metrics.json")) as f:
            self.precomputed_metrics = json.load(f)

        with open(os.path.join(DATA_DIR, "similarity_matrices.json")) as f:
            self.similarity_matrices = json.load(f)

        with open(os.path.join(DATA_DIR, "tfidf_top_terms.json")) as f:
            self.tfidf_top_terms = json.load(f)

        # ── Build lookup index ────────────────────────────────────────────────
        self._books_by_id = {b["book_id"]: b for b in self.books_catalog}

    def get_book(self, book_id) -> dict:
        return self._books_by_id.get(book_id) or self._books_by_id.get(str(book_id)) or {}

    def models_loaded_status(self) -> Dict[str, bool]:
        return {
            "cf_user_user": self.cf_uu is not None,
            "cf_item_item": self.cf_ii is not None,
            "content_based": self.content_based is not None,
            "hybrid": self.hybrid is not None,
        }

    def dataset_stats(self) -> Dict[str, int]:
        return {
            "books": len(self.books_catalog),
            "users": len(self.users_catalog),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
store = ModelStore()


def get_store() -> ModelStore:
    """FastAPI dependency — returns the loaded model store."""
    return store
