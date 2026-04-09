import type { Algorithm } from "./types";

export const ALGORITHMS: {
  id: Algorithm;
  label: string;
  shortLabel: string;
  description: string;
  detail: string;
  color: string;
  icon: string;
}[] = [
  {
    id: "cf-user",
    label: "Collaborative Filtering (User-User)",
    shortLabel: "CF User-User",
    description: "Finds users with similar taste and recommends what they liked.",
    detail:
      "Builds a user–item rating matrix, computes cosine similarity between users, then predicts ratings via a weighted average of the k-nearest neighbours. Works best for active users with many ratings.",
    color: "text-blue-400",
    icon: "Users",
  },
  {
    id: "cf-item",
    label: "Collaborative Filtering (Item-Item)",
    shortLabel: "CF Item-Item",
    description: "Recommends books similar to the ones you've already rated.",
    detail:
      "Transposes the user–item matrix to compute item–item cosine similarity. More stable than user-user CF on sparse data and scales better to large catalogs.",
    color: "text-purple-400",
    icon: "BookOpen",
  },
  {
    id: "content",
    label: "Content-Based",
    shortLabel: "Content-Based",
    description: "Matches books by title, author, and genre tags using TF-IDF.",
    detail:
      "Builds a TF-IDF feature vector from title + author + tags for each book, then uses cosine similarity via k-NN to find books with the most similar textual profile to your reading history.",
    color: "text-emerald-400",
    icon: "Tag",
  },
  {
    id: "hybrid",
    label: "Hybrid",
    shortLabel: "Hybrid",
    description: "Adaptively blends CF and content signals based on your history.",
    detail:
      "Dynamically adjusts the CF/CB weight based on how many ratings the user has. New users get mostly content-based; power users get mostly collaborative. You can drag the slider to override the weights.",
    color: "text-amber-400",
    icon: "Blend",
  },
];

export const METRIC_DESCRIPTIONS: Record<
  string,
  { label: string; formula: string; description: string; range: string }
> = {
  rmse: {
    label: "RMSE",
    formula: "√( Σ(r̂ᵢ − rᵢ)² / n )",
    description: "Root Mean Squared Error — penalises large rating prediction errors more than MAE. Lower is better.",
    range: "0 → ∞ (lower = better)",
  },
  mae: {
    label: "MAE",
    formula: "Σ|r̂ᵢ − rᵢ| / n",
    description: "Mean Absolute Error — average absolute difference between predicted and actual ratings. Lower is better.",
    range: "0 → ∞ (lower = better)",
  },
  "precision@10": {
    label: "Precision@10",
    formula: "|Relevant ∩ Top-10| / 10",
    description: "Fraction of the top-10 recommendations that are actually relevant (rated ≥ 3.0 in the test set).",
    range: "0 → 1 (higher = better)",
  },
  "recall@10": {
    label: "Recall@10",
    formula: "|Relevant ∩ Top-10| / |Relevant|",
    description: "Fraction of all relevant test items that appear in the top-10 recommendations.",
    range: "0 → 1 (higher = better)",
  },
  coverage: {
    label: "Coverage",
    formula: "|Recommended items| / |All items|",
    description: "Percentage of the catalog that appears in recommendations across all users. Higher coverage = less filter bubble.",
    range: "0 → 1 (higher = better)",
  },
  diversity: {
    label: "Diversity",
    formula: "avg pairwise(1 − sim(i, j))",
    description: "Average pairwise dissimilarity within recommendation lists. Higher diversity = more varied suggestions.",
    range: "0 → 1 (higher = better)",
  },
};
