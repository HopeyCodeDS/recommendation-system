"""
Command-line interface for the recommender system
"""

import argparse
import yaml
import pandas as pd
import sys
import os

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from src.data.data_loader import DataLoader
from src.recommenders.collaborative_filter import CollaborativeFilter
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.evaluation.evaluator import RecommenderEvaluator


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_recommender(recommender_type, config, ratings_df, books_df=None):
    """Train a recommender based on type."""
    if recommender_type == 'collaborative':
        cf_config = config.get('collaborative_filtering', {})
        recommender = CollaborativeFilter(
            method=cf_config.get('method', 'user-user'),
            k_neighbors=cf_config.get('k_neighbors', 50),
            normalize_ratings=cf_config.get('normalize_ratings', True)
        )
        recommender.fit(ratings_df)
    
    elif recommender_type == 'content':
        cb_config = config.get('content_based', {})
        recommender = ContentBasedRecommender(
            max_features=cb_config.get('max_features', 5000),
            n_neighbors=cb_config.get('n_neighbors', 10)
        )
        recommender.fit(ratings_df, books_df)
    
    elif recommender_type == 'hybrid':
        hybrid_config = config.get('hybrid', {})
        recommender = HybridRecommender(
            cf_weight=hybrid_config.get('cf_weight', 0.6),
            cb_weight=hybrid_config.get('cb_weight', 0.4),
            adaptive_weighting=hybrid_config.get('adaptive_weighting', True)
        )
        recommender.fit(ratings_df, books_df)
    
    else:
        raise ValueError(f"Unknown recommender type: {recommender_type}")
    
    return recommender


def main():
    parser = argparse.ArgumentParser(description='Book Recommender System CLI')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--type', type=str, choices=['collaborative', 'content', 'hybrid'],
                       default='hybrid', help='Type of recommender to use')
    parser.add_argument('--user-id', type=str, help='User ID to get recommendations for')
    parser.add_argument('--n-recommendations', type=int, default=10,
                       help='Number of recommendations to return')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the recommender on test data')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all recommender types')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {args.config}. Using defaults.")
        config = {}
    
    # Load data
    print("Loading data...")
    loader = DataLoader()
    
    try:
        ratings_df = loader.load_ratings()
        books_df = loader.load_books()
        loader.load_tags()
        
        # Preprocess
        preprocess_config = config.get('preprocessing', {})
        processed_ratings = loader.preprocess_ratings(
            min_user_ratings=preprocess_config.get('min_user_ratings', 3),
            min_book_ratings=preprocess_config.get('min_item_ratings', 3)
        )
        processed_books = loader.merge_books_with_tags()
        
        print(f"Loaded {len(processed_ratings)} ratings and {len(processed_books)} books")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Evaluate or compare
    if args.compare:
        print("\nComparing all recommender types...")
        evaluator = RecommenderEvaluator(
            test_size=config.get('evaluation', {}).get('test_size', 0.2)
        )
        train_data, test_data = evaluator.train_test_split(processed_ratings)
        
        recommenders = {
            'Collaborative': CollaborativeFilter(),
            'Content-Based': ContentBasedRecommender(),
            'Hybrid': HybridRecommender()
        }
        
        # Train all
        for name, rec in recommenders.items():
            print(f"Training {name}...")
            if name == 'Content-Based' or name == 'Hybrid':
                rec.fit(train_data, processed_books)
            else:
                rec.fit(train_data)
        
        # Compare
        comparison = evaluator.compare_recommenders(
            recommenders,
            test_data=test_data,
            metrics=config.get('evaluation', {}).get('metrics', ['precision', 'recall', 'rmse'])
        )
        
        print("\nComparison Results:")
        print(comparison)
        return
    
    if args.evaluate:
        print(f"\nEvaluating {args.type} recommender...")
        evaluator = RecommenderEvaluator(
            test_size=config.get('evaluation', {}).get('test_size', 0.2)
        )
        train_data, test_data = evaluator.train_test_split(processed_ratings)
        
        recommender = train_recommender(args.type, config, train_data, processed_books)
        
        results = evaluator.evaluate(
            recommender,
            test_data=test_data,
            train_data=train_data,
            metrics=config.get('evaluation', {}).get('metrics', ['precision', 'recall', 'rmse'])
        )
        
        evaluator.print_results(results)
        return
    
    # Get recommendations
    if args.user_id:
        print(f"\nTraining {args.type} recommender...")
        recommender = train_recommender(args.type, config, processed_ratings, processed_books)

        target_user_id = args.user_id
        if 'user_id' in processed_ratings.columns and not processed_ratings.empty:
            user_dtype = processed_ratings['user_id'].dtype
            try:
                if pd.api.types.is_integer_dtype(user_dtype):
                    target_user_id = int(args.user_id)
                elif pd.api.types.is_float_dtype(user_dtype):
                    target_user_id = float(args.user_id)
                else:
                    target_user_id = str(args.user_id)
            except ValueError:
                print(f"Invalid --user-id '{args.user_id}' for dtype {user_dtype}")
                return
            
        print(f"\nGetting recommendations for user {target_user_id}...")
        recommendations = recommender.recommend(
            target_user_id,
            n_recommendations=args.n_recommendations
        )
        
        print(f"\nTop {len(recommendations)} Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['item_id']}")
            print(f"   Predicted Rating: {rec['predicted_rating']:.2f}")
            if 'confidence' in rec:
                print(f"   Confidence: {rec['confidence']:.2f}")
            print()
    else:
        print("Please provide --user-id to get recommendations, or use --evaluate or --compare")


if __name__ == '__main__':
    main()

