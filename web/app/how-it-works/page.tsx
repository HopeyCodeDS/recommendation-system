import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SimilarityHeatmap } from "@/components/how-it-works/SimilarityHeatmap";
import { TFIDFVisualizer } from "@/components/how-it-works/TFIDFVisualizer";
import { api } from "@/lib/api";

export const metadata = {
  title: "How It Works — BookRec",
  description: "Technical walkthrough of the CF, Content-Based, and Hybrid algorithms.",
};

const CF_CODE = `class CollaborativeFilter:
    def fit(self, ratings_df):
        # Build sparse user-item matrix
        self.user_item_matrix, self.user_mapping, self.item_mapping = (
            create_user_item_matrix(filtered_df, item_col=self.item_col)
        )
        # Compute cosine similarity between all user pairs
        self.similarity_matrix = cosine_similarity_sparse(
            self.user_item_matrix, dense_output=True
        )

    def predict(self, user_id, item_id):
        # Weighted average of k-nearest neighbours' ratings
        k_neighbors = sorted(
            enumerate(self.similarity_matrix[user_idx]),
            key=lambda x: x[1], reverse=True
        )[:self.k_neighbors]
        # …
        return float(np.clip(prediction + user_mean, 1.0, 5.0))`;

const CB_CODE = `class ContentBasedRecommender:
    def fit(self, ratings_df, books_df):
        # Combine title + authors + tags into one content string
        self.books_df['content'] = (
            books_df['title'] + ' ' +
            books_df['authors'] + ' ' +
            books_df['tags']
        )
        # TF-IDF vectorisation
        self.tfidf_matrix = TfidfVectorizer(
            stop_words='english', max_features=500
        ).fit_transform(self.books_df['content'])
        # KNN index over TF-IDF vectors (cosine metric)
        self.knn_model = NearestNeighbors(metric='cosine').fit(self.tfidf_matrix)

    def predict(self, user_id, item_id):
        # Similarity-weighted average of user's past ratings
        sims = cosine_similarity(target_vec, rated_matrix).flatten()
        return float(np.clip(
            np.dot(sims, ratings) / np.sum(np.abs(sims)), 1.0, 5.0
        ))`;

const HYBRID_CODE = `class HybridRecommender:
    def _get_weights(self, user_id):
        n = len(self.ratings_df[self.ratings_df['user_id'] == user_id])
        if n < 2:
            return 0.2, 0.8   # cold-start → mostly content-based
        elif n < 10:
            # linearly interpolate
            t = (n - 2) / 8
            return 0.2 + 0.5 * t, 0.8 - 0.5 * t
        else:
            return 0.7, 0.3   # power user → mostly collaborative

    def recommend(self, user_id, n_recommendations=10):
        cf_w, cb_w = self._get_weights(user_id)
        cf_recs = {r['item_id']: r for r in self.cf_recommender.recommend(...)}
        cb_recs = {r['item_id']: r for r in self.cb_recommender.recommend(...)}
        all_items = set(cf_recs) | set(cb_recs)
        scores = {
            item: cf_w * cf_recs.get(item, neutral).predicted_rating
                + cb_w * cb_recs.get(item, neutral).predicted_rating
            for item in all_items
        }
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)`;

interface CodeBlockProps {
  code: string;
  title: string;
}

function CodeBlock({ code, title }: CodeBlockProps) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950 overflow-hidden">
      <div className="border-b border-zinc-800 px-4 py-2 text-xs font-mono text-zinc-400">
        {title}
      </div>
      <pre className="overflow-x-auto p-4 text-xs leading-relaxed text-zinc-300">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export default async function HowItWorksPage() {
  const [simMatrix, books, tfidfTerms] = await Promise.all([
    api.similarity.users().catch(() => ({ users: [], matrix: [] })),
    api.books.list().catch(() => []),
    api.tfidf
      .terms(2)
      .catch(() => []),
  ]);

  // Books that have TF-IDF data (have catalog metadata)
  const catalogBooks = books.filter((b) => !b.title.startsWith("Book #")).slice(0, 30);
  const firstBook = catalogBooks[0];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">How It Works</h1>
        <p className="mt-2 text-zinc-400">
          A technical walkthrough of each algorithm — from data to predictions.
        </p>
      </div>

      <Tabs defaultValue="cf">
        <TabsList className="bg-zinc-900 border border-zinc-800">
          <TabsTrigger value="cf">Collaborative Filtering</TabsTrigger>
          <TabsTrigger value="cb">Content-Based</TabsTrigger>
          <TabsTrigger value="hybrid">Hybrid</TabsTrigger>
        </TabsList>

        {/* CF Tab */}
        <TabsContent value="cf" className="mt-6 space-y-6">
          <div className="space-y-2">
            <h2 className="text-xl font-semibold text-white">Collaborative Filtering</h2>
            <p className="text-zinc-400">
              Builds a sparse user–item rating matrix and computes cosine similarity between
              every pair of users. To predict a rating, it takes a weighted average of the
              k-nearest neighbours&apos; ratings for that item.
            </p>
          </div>

          {simMatrix.users.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-zinc-300">
                User-User Similarity Matrix (from trained model)
              </h3>
              <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
                <SimilarityHeatmap data={simMatrix} />
              </div>
            </div>
          )}

          <CodeBlock code={CF_CODE} title="src/recommenders/collaborative_filter.py (key excerpts)" />
        </TabsContent>

        {/* Content-Based Tab */}
        <TabsContent value="cb" className="mt-6 space-y-6">
          <div className="space-y-2">
            <h2 className="text-xl font-semibold text-white">Content-Based Filtering</h2>
            <p className="text-zinc-400">
              Concatenates title + author + genre tags for each book and runs TF-IDF
              vectorisation. A KNN index (cosine distance) finds books with the most similar
              feature vectors to a user&apos;s profile (average of their rated books&apos; TF-IDF vectors).
            </p>
          </div>

          {catalogBooks.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-zinc-300">
                TF-IDF Term Importance — select a book
              </h3>
              <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
                <TFIDFVisualizer
                  books={catalogBooks}
                  initialTerms={tfidfTerms}
                  initialBookId={firstBook?.book_id ?? 2}
                />
              </div>
            </div>
          )}

          <CodeBlock code={CB_CODE} title="src/recommenders/content_based.py (key excerpts)" />
        </TabsContent>

        {/* Hybrid Tab */}
        <TabsContent value="hybrid" className="mt-6 space-y-6">
          <div className="space-y-2">
            <h2 className="text-xl font-semibold text-white">Hybrid Recommender</h2>
            <p className="text-zinc-400">
              Adaptively blends the CF and Content-Based scores. New users (&lt;2 ratings) get
              80% CB / 20% CF. Power users (≥10 ratings) get 70% CF / 30% CB. The weight
              interpolates linearly in between. The demo slider lets you override this.
            </p>
          </div>

          {/* Weight diagram */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
            <h3 className="mb-4 text-sm font-semibold text-zinc-300">
              Adaptive Weight Shift
            </h3>
            <div className="space-y-3">
              {[
                { label: "0 ratings (cold start)", cf: 20, cb: 80 },
                { label: "2 ratings", cf: 20, cb: 80 },
                { label: "5 ratings", cf: 39, cb: 61 },
                { label: "8 ratings", cf: 58, cb: 42 },
                { label: "10+ ratings", cf: 70, cb: 30 },
              ].map(({ label, cf, cb }) => (
                <div key={label} className="flex items-center gap-3 text-xs">
                  <span className="w-36 text-zinc-500 shrink-0">{label}</span>
                  <div className="flex-1 flex rounded overflow-hidden h-5">
                    <div
                      className="bg-blue-500/60 flex items-center justify-center text-white font-mono"
                      style={{ width: `${cf}%` }}
                    >
                      CF {cf}%
                    </div>
                    <div
                      className="bg-emerald-500/60 flex items-center justify-center text-white font-mono"
                      style={{ width: `${cb}%` }}
                    >
                      CB {cb}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <CodeBlock code={HYBRID_CODE} title="src/recommenders/hybrid_recommender.py (key excerpts)" />
        </TabsContent>
      </Tabs>
    </div>
  );
}
