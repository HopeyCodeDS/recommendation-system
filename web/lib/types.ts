// Mirrors Pydantic schemas from api/schemas.py

export interface Book {
  book_id: number;
  title: string;
  authors: string | null;
  average_rating: number | null;
  image_url: string | null;
  small_image_url: string | null;
  original_publication_year: number | null;
  language_code: string | null;
  tags: string | null;
}

export interface BookDetail extends Book {
  similar_books: SimilarBook[];
}

export interface SimilarBook {
  book_id: number | string;
  title: string | null;
  authors: string | null;
  image_url: string | null;
  similarity_score: number;
}

export interface User {
  user_id: number | string;
  display_name: string;
  rating_count: number;
  avg_rating: number;
  profile_type: "power_user" | "active" | "casual";
}

export type Algorithm = "cf-user" | "cf-item" | "content" | "hybrid";

export interface RecommendationItem {
  rank: number;
  item_id: number | string;
  title: string | null;
  authors: string | null;
  image_url: string | null;
  average_rating: number | null;
  predicted_rating: number;
  confidence: number | null;
  explanation: string;
}

export interface RecommendationResponse {
  user_id: number | string;
  algorithm: string;
  recommendations: RecommendationItem[];
}

export interface MetricsData {
  [recommender: string]: {
    rmse?: number;
    mae?: number;
    "precision@5"?: number;
    "precision@10"?: number;
    "precision@20"?: number;
    "recall@5"?: number;
    "recall@10"?: number;
    "recall@20"?: number;
    "f1@5"?: number;
    "f1@10"?: number;
    "f1@20"?: number;
    coverage?: number;
    diversity?: number;
    novelty?: number;
  };
}

export interface SimilarityMatrix {
  users: string[];
  matrix: number[][];
}

export interface TFIDFTerm {
  term: string;
  score: number;
}
