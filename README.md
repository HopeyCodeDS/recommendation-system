# Hybrid Recommender System

A comprehensive book recommendation system that combines collaborative filtering and content-based filtering approaches. This project provides a production-ready implementation with proper architecture, evaluation metrics, and documentation.

## Features

- **Collaborative Filtering**: User-user and item-item collaborative filtering with configurable similarity metrics
- **Content-Based Filtering**: TF-IDF-based content filtering using book metadata (title, authors, tags)
- **Hybrid Approach**: Adaptive combination of both methods with intelligent weighting
- **Comprehensive Evaluation**: Multiple metrics including Precision@K, Recall@K, RMSE, MAE, Coverage, Diversity
- **Cold-Start Handling**: Handles new users and new items gracefully
- **Modular Architecture**: Clean, extensible codebase following best practices

## Project Structure

```
Recommender_Systems/
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── recommenders/       # Recommender implementations
│   ├── evaluation/         # Evaluation metrics and framework
│   └── utils/             # Utility functions
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit tests
├── config/                 # Configuration files
├── data/                   # Data files (raw and processed)
├── main.py                 # CLI interface
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Recommender_Systems
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your data files are in the correct locations (see Configuration section)

## Quick Start

### Using the CLI

Get recommendations for a user:
```bash
python main.py --type hybrid --user-id USER_ID --n-recommendations 10
```

Evaluate a recommender:
```bash
python main.py --type hybrid --evaluate
```

Compare all recommender types:
```bash
python main.py --compare
```

### Using Python API

```python
from src.data.data_loader import DataLoader
from src.recommenders.hybrid_recommender import HybridRecommender

# Load data
loader = DataLoader()
ratings_df = loader.load_ratings()
books_df = loader.load_books()
loader.load_tags()

# Preprocess
processed_ratings = loader.preprocess_ratings(min_user_ratings=3, min_book_ratings=3)
processed_books = loader.merge_books_with_tags()

# Train hybrid recommender
recommender = HybridRecommender(cf_weight=0.6, cb_weight=0.4, adaptive_weighting=True)
recommender.fit(processed_ratings, processed_books)

# Get recommendations
recommendations = recommender.recommend('user_id', n_recommendations=10)
for rec in recommendations:
    print(f"{rec['item_id']}: {rec['predicted_rating']:.2f}")
```

## Configuration

Edit `config/config.yaml` to customize:

- Data paths
- Preprocessing parameters (minimum ratings per user/item)
- Collaborative filtering settings (method, k-neighbors, similarity metric)
- Content-based settings (TF-IDF parameters)
- Hybrid recommender weights
- Evaluation settings

## Data Format

The system expects the following data files:

### Ratings (`ratings.csv`)
- `user_id`: User identifier
- `book_id` or `title`: Item identifier
- `rating`: Rating value (typically 1-5)

### Books (`books.csv`)
- `book_id`: Book identifier
- `title`: Book title
- `authors`: Author names
- `average_rating`: Average rating (optional)

### Tags (`tags.csv`, `book_tags.csv`)
- Tag metadata for content-based filtering

## Recommender Types

### Collaborative Filtering
- **User-User**: Finds similar users and recommends items they liked
- **Item-Item**: Finds similar items based on user ratings
- Supports cosine similarity and Pearson correlation
- Handles sparse data efficiently using sparse matrices

### Content-Based Filtering
- Uses TF-IDF vectorization on book metadata
- Combines title, authors, and tags
- Finds similar items based on content features
- Excellent for cold-start scenarios

### Hybrid Recommender
- Combines collaborative and content-based approaches
- Adaptive weighting based on user history:
  - New users (< 10 ratings): More weight on content-based (80%)
  - Active users (≥ 10 ratings): More weight on collaborative (70%)
- Configurable blending strategies

## Evaluation Metrics

The system supports multiple evaluation metrics:

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **F1@K**: Harmonic mean of precision and recall
- **RMSE**: Root Mean Squared Error for rating prediction
- **MAE**: Mean Absolute Error for rating prediction
- **Coverage**: Fraction of catalog that can be recommended
- **Diversity**: Average dissimilarity of recommendations
- **Novelty**: Average negative log popularity

## Examples

See the Jupyter notebooks in the `notebooks/` directory:

- `01_data_exploration.ipynb`: Explore and visualize the dataset
- `04_hybrid_recommender.ipynb`: Train and use the hybrid recommender

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

Run specific test file:
```bash
python -m pytest tests/test_hybrid.py
```

## API Documentation

### BaseRecommender

Base class for all recommenders with common interface:

- `fit(ratings_df)`: Train the recommender
- `predict(user_id, item_id)`: Predict rating for user-item pair
- `recommend(user_id, n_recommendations)`: Get top N recommendations
- `evaluate(test_ratings_df)`: Evaluate on test data

### CollaborativeFilter

```python
from src.recommenders.collaborative_filter import CollaborativeFilter

cf = CollaborativeFilter(
    method='user-user',  # or 'item-item'
    similarity_metric='cosine',  # or 'pearson'
    k_neighbors=50,
    normalize_ratings=True
)
cf.fit(ratings_df)
```

### ContentBasedRecommender

```python
from src.recommenders.content_based import ContentBasedRecommender

cb = ContentBasedRecommender(
    max_features=5000,
    n_neighbors=10,
    use_tags=True,
    use_authors=True
)
cb.fit(ratings_df, books_df)
```

### HybridRecommender

```python
from src.recommenders.hybrid_recommender import HybridRecommender

hybrid = HybridRecommender(
    cf_weight=0.6,
    cb_weight=0.4,
    adaptive_weighting=True
)
hybrid.fit(ratings_df, books_df)
```

## Performance Considerations

- Uses sparse matrices for memory efficiency with large datasets
- Precomputes similarity matrices for faster recommendations
- Configurable filtering to reduce data sparsity
- Supports incremental updates (similarity matrices can be cached)

## Contributing

1. Follow the existing code structure
2. Add unit tests for new features
3. Update documentation
4. Ensure code passes linting

## License

[Your License Here]

## Acknowledgments

This project implements standard recommender system algorithms and best practices from the research literature.

