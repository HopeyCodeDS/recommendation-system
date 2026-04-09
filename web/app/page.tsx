import { HeroSection } from "@/components/landing/HeroSection";
import { AlgorithmCards } from "@/components/landing/AlgorithmCards";
import { MetricsPreview } from "@/components/landing/MetricsPreview";
import { api } from "@/lib/api";

export default async function HomePage() {
  let metrics = {};
  try {
    metrics = await api.metrics.comparison();
  } catch {
    // API may not be available at build time; MetricsPreview handles empty gracefully
  }

  return (
    <div className="space-y-6">
      <HeroSection />
      <AlgorithmCards />
      <MetricsPreview metrics={metrics} />

      <section className="pt-6 text-center">
        <p className="text-sm text-zinc-500">
          Stack:{" "}
          {[
            "Python 3.11",
            "scikit-learn",
            "pandas",
            "FastAPI",
            "Next.js 16",
            "shadcn/ui",
            "Recharts",
            "Framer Motion",
          ].map((t, i) => (
            <span key={t}>
              <span className="font-mono text-zinc-400">{t}</span>
              {i < 7 && <span className="mx-2 text-zinc-700">·</span>}
            </span>
          ))}
        </p>
      </section>
    </div>
  );
}
