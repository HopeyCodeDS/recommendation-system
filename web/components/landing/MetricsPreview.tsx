import type { MetricsData } from "@/lib/types";

interface Props {
  metrics: MetricsData;
}

export function MetricsPreview({ metrics }: Props) {
  const hybrid = metrics["Hybrid"] ?? {};
  const cfUU = metrics["CF User-User"] ?? {};

  const stats = [
    {
      label: "RMSE (Hybrid)",
      value: hybrid.rmse != null ? hybrid.rmse.toFixed(3) : "—",
      note: "rating prediction error",
    },
    {
      label: "RMSE (CF)",
      value: cfUU.rmse != null ? cfUU.rmse.toFixed(3) : "—",
      note: "vs. content-based",
    },
    {
      label: "Algorithms",
      value: Object.keys(metrics).length.toString(),
      note: "evaluated & compared",
    },
    {
      label: "Books",
      value: "99",
      note: "Goodreads catalog",
    },
  ];

  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 py-10">
      <h2 className="mb-8 text-center text-xl font-semibold text-white">
        Evaluated on a holdout test set (80/20 split)
      </h2>
      <div className="grid grid-cols-2 gap-6 px-8 sm:grid-cols-4">
        {stats.map((s) => (
          <div key={s.label} className="text-center">
            <div className="text-3xl font-bold text-amber-400">{s.value}</div>
            <div className="mt-1 text-sm font-medium text-zinc-300">{s.label}</div>
            <div className="text-xs text-zinc-500">{s.note}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
