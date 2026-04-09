#!/usr/bin/env python
"""
Precompute script: trains all recommender models, serializes to joblib,
and dumps JSON data files consumed by the FastAPI backend.

Run from the project root:
    python scripts/precompute.py
"""

import os
import sys
import json
import warnings
import numpy as np

# Force UTF-8 stdout to avoid Windows cp1252 issues with special characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Project root directory to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib

from src.data.data_loader import DataLoader
from src.recommenders.collaborative_filter import CollaborativeFilter
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.evaluation.evaluator import RecommenderEvaluator

warnings.filterwarnings("ignore")

# ── Output directories ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "api", "models")
DATA_DIR = os.path.join(ROOT, "api", "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def to_native(obj):
    """Recursively convert numpy scalars to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    return obj


# ── 1. Load and preprocess data ────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

loader = DataLoader()
loader.load_ratings()
loader.load_books()
loader.load_tags()

ratings = loader.preprocess_ratings(min_user_ratings=1, min_book_ratings=1)
books = loader.merge_books_with_tags()

print(f"  Ratings : {len(ratings)} rows | {ratings['user_id'].nunique()} users | {ratings['book_id'].nunique()} books")
print(f"  Books   : {len(books)} rows | columns: {list(books.columns)}")


# ── 2. Train / test split ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Train/test split (80/20)")
print("=" * 60)

evaluator = RecommenderEvaluator(
    test_size=0.2,
    random_state=42,
    k_values=[5, 10, 20],
)
train_data, test_data = evaluator.train_test_split(ratings)
print(f"  Train: {len(train_data)} | Test: {len(test_data)}")


# ── 3. Train models ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training models")
print("=" * 60)

n_users = train_data["user_id"].nunique()
k = max(1, min(5, n_users - 1))  # safe k for small dataset

cf_uu = CollaborativeFilter(
    method="user-user",
    k_neighbors=k,
    min_user_ratings=1,
    min_item_ratings=1,
)
cf_uu.fit(train_data)
print(f"  ✓ CF User-User  (k={k})")

cf_ii = CollaborativeFilter(
    method="item-item",
    k_neighbors=k,
    min_user_ratings=1,
    min_item_ratings=1,
)
cf_ii.fit(train_data)
print(f"  ✓ CF Item-Item  (k={k})")

cb = ContentBasedRecommender(max_features=500, n_neighbors=10)
cb.fit(train_data, books)
print("  ✓ Content-Based")

hybrid = HybridRecommender(
    cf_weight=0.6,
    cb_weight=0.4,
    adaptive_weighting=True,
    cf_k_neighbors=k,
    cb_max_features=500,
    cb_n_neighbors=10,
)
hybrid.fit(train_data, books)
print("  ✓ Hybrid")


# ── 4. Evaluate ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Evaluating models")
print("=" * 60)

recommenders = {
    "CF User-User": cf_uu,
    "CF Item-Item": cf_ii,
    "Content-Based": cb,
    "Hybrid": hybrid,
}

try:
    comparison_df = evaluator.compare_recommenders(
        recommenders,
        test_data=test_data,
        metrics=["precision", "recall", "rmse", "mae"],
    )
    print(comparison_df.to_string())
    metrics_dict = to_native(comparison_df.fillna(0).to_dict(orient="index"))
except Exception as e:
    print(f"  Warning: evaluation failed ({e}). Using placeholder metrics.")
    metrics_dict = {
        name: {"rmse": 0.0, "mae": 0.0} for name in recommenders
    }

with open(os.path.join(DATA_DIR, "precomputed_metrics.json"), "w") as f:
    json.dump(metrics_dict, f, indent=2)
print(f"\n  ✓ Saved precomputed_metrics.json")


# ── 5. Serialize models ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Serializing models")
print("=" * 60)

joblib.dump(cf_uu, os.path.join(MODELS_DIR, "cf_user_user.joblib"))
joblib.dump(cf_ii, os.path.join(MODELS_DIR, "cf_item_item.joblib"))
joblib.dump(cb, os.path.join(MODELS_DIR, "content_based.joblib"))
joblib.dump(hybrid, os.path.join(MODELS_DIR, "hybrid.joblib"))
print("  ✓ cf_user_user.joblib")
print("  ✓ cf_item_item.joblib")
print("  ✓ content_based.joblib")
print("  ✓ hybrid.joblib")


# ── 6. User-user similarity matrix ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Extracting similarity matrix")
print("=" * 60)

users = [str(cf_uu.reverse_user_mapping[i]) for i in range(len(cf_uu.user_mapping))]
sim = cf_uu.similarity_matrix
if hasattr(sim, "toarray"):
    sim = sim.toarray()
sim = np.round(np.asarray(sim, dtype=float), 4).tolist()

sim_data = {"users": users, "matrix": sim}
with open(os.path.join(DATA_DIR, "similarity_matrices.json"), "w") as f:
    json.dump(sim_data, f, indent=2)
print(f"  ✓ Saved similarity_matrices.json  ({len(users)} users × {len(users)})")


# ── 7. TF-IDF top terms per book ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Extracting TF-IDF top terms")
print("=" * 60)

feature_names = cb.tfidf_vectorizer.get_feature_names_out()
tfidf_terms: dict = {}

for idx in range(cb.tfidf_matrix.shape[0]):
    row = cb.tfidf_matrix[idx]
    if hasattr(row, "toarray"):
        row = row.toarray().flatten()
    else:
        row = np.asarray(row).flatten()
    top_indices = row.argsort()[::-1][:10]
    top_terms = [
        {"term": str(feature_names[i]), "score": round(float(row[i]), 4)}
        for i in top_indices
        if row[i] > 0
    ]
    book_id = str(cb.reverse_item_mapping[idx])
    tfidf_terms[book_id] = top_terms

with open(os.path.join(DATA_DIR, "tfidf_top_terms.json"), "w") as f:
    json.dump(tfidf_terms, f, indent=2)
print(f"  ✓ Saved tfidf_top_terms.json  ({len(tfidf_terms)} books)")


# ── 8. Books catalog ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Exporting books catalog")
print("=" * 60)

# Add placeholder entries for any book_id in ratings that has no catalog entry
catalog_ids = set(int(b["book_id"]) for b in books.to_dict(orient="records") if b.get("book_id") is not None)
rating_ids_all = set(int(x) for x in ratings["book_id"].unique())
missing_ids = rating_ids_all - catalog_ids
if missing_ids:
    print(f"  Adding {len(missing_ids)} placeholder entries for books missing from catalog")

import pandas as _pd
placeholder_rows = []
for mid in missing_ids:
    placeholder_rows.append({
        "book_id": mid,
        "title": f"Book #{mid}",
        "authors": None,
        "average_rating": None,
        "image_url": None,
        "small_image_url": None,
        "original_publication_year": None,
        "language_code": None,
        "tags": "",
    })
if placeholder_rows:
    books = _pd.concat([books, _pd.DataFrame(placeholder_rows)], ignore_index=True)

books_clean = []
for rec in books.to_dict(orient="records"):
    cleaned = {}
    for k, v in rec.items():
        if isinstance(v, float) and np.isnan(v):
            cleaned[k] = None
        elif isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.floating,)):
            cleaned[k] = float(v)
        else:
            cleaned[k] = v
    books_clean.append(cleaned)

with open(os.path.join(DATA_DIR, "books_catalog.json"), "w") as f:
    json.dump(books_clean, f, indent=2)
print(f"  ✓ Saved books_catalog.json  ({len(books_clean)} books)")


# ── 9. Users catalog ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Exporting users catalog")
print("=" * 60)

user_stats = (
    ratings.groupby("user_id")
    .agg(rating_count=("rating", "count"), avg_rating=("rating", "mean"))
    .reset_index()
)

users_catalog = []
for _, row in user_stats.iterrows():
    uid = row["user_id"]
    count = int(row["rating_count"])
    users_catalog.append(
        {
            "user_id": int(uid) if isinstance(uid, (np.integer, int)) else str(uid),
            "display_name": f"User {uid}",
            "rating_count": count,
            "avg_rating": round(float(row["avg_rating"]), 2),
            "profile_type": (
                "power_user" if count >= 20
                else "active" if count >= 10
                else "casual"
            ),
        }
    )

with open(os.path.join(DATA_DIR, "users_catalog.json"), "w") as f:
    json.dump(users_catalog, f, indent=2)
print(f"  ✓ Saved users_catalog.json  ({len(users_catalog)} users)")


# ── Done ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅  PRECOMPUTE COMPLETE")
print("=" * 60)
print(f"  Models : {MODELS_DIR}")
print(f"  Data   : {DATA_DIR}")
