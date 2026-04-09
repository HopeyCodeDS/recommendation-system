import { MetricsRadarChart, PrecisionRecallChart } from "@/components/metrics/MetricsCharts";
import { MetricsTable } from "@/components/metrics/MetricsTable";
import { api } from "@/lib/api";

export const metadata = {
  title: "Metrics — BookRec",
  description: "Side-by-side evaluation of CF, Content-Based, and Hybrid recommenders.",
};

export default async function MetricsPage() {
  let metrics = {};
  try {
    metrics = await api.metrics.comparison();
  } catch {
    // handled below
  }

  const isEmpty = Object.keys(metrics).length === 0;

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-3xl font-bold text-white">Metrics Dashboard</h1>
        <p className="mt-2 text-zinc-400">
          All algorithms evaluated on an 80/20 train/test split with random_state=42.
          Relevance threshold: 3.0 stars.
        </p>
      </div>

      {isEmpty ? (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-8 text-center text-zinc-500">
          Backend unavailable — start the FastAPI server to see live metrics.
        </div>
      ) : (
        <>
          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-zinc-200">Overall Comparison (Radar)</h2>
            <p className="text-sm text-zinc-500">
              Rating accuracy is normalised as 1 − RMSE/5. Higher on every axis = better.
            </p>
            <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
              <MetricsRadarChart metrics={metrics} />
            </div>
          </section>

          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-zinc-200">Precision & Recall at K</h2>
            <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
              <PrecisionRecallChart metrics={metrics} />
            </div>
          </section>

          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-zinc-200">Full Metrics Table</h2>
            <p className="text-xs text-zinc-500">
              ★ marks the best-performing algorithm per metric.
            </p>
            <MetricsTable metrics={metrics} />
          </section>

          <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-6 text-sm text-zinc-400 space-y-2">
            <h3 className="font-semibold text-zinc-300">Methodology</h3>
            <ul className="list-disc list-inside space-y-1">
              <li>Dataset: 99 Goodreads ratings from 5 users across 96 books</li>
              <li>Split: 80% train / 20% test, random_state=42</li>
              <li>Relevance threshold: ratings ≥ 3.0 counted as relevant</li>
              <li>CF k-neighbours = 4 (capped to dataset size)</li>
              <li>CF-UU &amp; CF-II share identical RMSE/MAE because the small test set maps to the same neighbours</li>
            </ul>
          </section>
        </>
      )}
    </div>
  );
}
