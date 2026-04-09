import type {
  Book,
  BookDetail,
  User,
  Algorithm,
  RecommendationResponse,
  MetricsData,
  SimilarityMatrix,
  TFIDFTerm,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8002";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  books: {
    list: () => get<Book[]>("/books"),
    get: (id: number) => get<BookDetail>(`/books/${id}`),
  },

  users: {
    list: () => get<User[]>("/users"),
  },

  recommendations: {
    get: (
      userId: number | string,
      algorithm: Algorithm,
      n = 10,
      cfWeight?: number,
    ) =>
      post<RecommendationResponse>("/recommendations", {
        user_id: userId,
        algorithm,
        n,
        cf_weight: cfWeight ?? null,
      }),

    similar: (bookId: number, n = 8) =>
      get<RecommendationResponse["recommendations"]>(
        `/recommendations/similar/${bookId}?n=${n}`,
      ),

    coldStart: (n = 10) => get<RecommendationResponse>(`/recommendations/cold-start?n=${n}`),
  },

  metrics: {
    comparison: () =>
      get<{ metrics: MetricsData }>("/metrics/comparison").then((r) => r.metrics),
  },

  similarity: {
    users: () => get<SimilarityMatrix>("/similarity/users"),
  },

  tfidf: {
    terms: (bookId: number | string) =>
      get<{ book_id: string; terms: TFIDFTerm[] }>(`/tfidf/${bookId}`).then(
        (r) => r.terms,
      ),
  },
};
