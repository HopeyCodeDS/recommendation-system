"use client";

import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { MetricsData } from "@/lib/types";

const COLORS = {
  "CF User-User": "#60a5fa",
  "CF Item-Item": "#a78bfa",
  "Content-Based": "#34d399",
  Hybrid: "#fbbf24",
};

interface Props {
  metrics: MetricsData;
}

export function MetricsRadarChart({ metrics }: Props) {
  const recommenders = Object.keys(metrics);

  const axes = [
    { key: "rmse_inv", label: "Rating Accuracy" },
    { key: "precision@10", label: "Precision@10" },
    { key: "recall@10", label: "Recall@10" },
    { key: "coverage", label: "Coverage" },
    { key: "diversity", label: "Diversity" },
  ];

  // Normalise RMSE to 0-1 accuracy (1 - rmse/5)
  const data = axes.map(({ key, label }) => {
    const point: Record<string, string | number> = { axis: label };
    recommenders.forEach((name) => {
      const m = metrics[name];
      if (key === "rmse_inv") {
        const rmse = m.rmse ?? 1;
        point[name] = parseFloat(Math.max(0, 1 - rmse / 5).toFixed(3));
      } else {
        point[name] = parseFloat(((m as Record<string, number>)[key] ?? 0).toFixed(3));
      }
    });
    return point;
  });

  return (
    <ResponsiveContainer width="100%" height={340}>
      <RadarChart data={data}>
        <PolarGrid stroke="#3f3f46" />
        <PolarAngleAxis dataKey="axis" tick={{ fill: "#a1a1aa", fontSize: 12 }} />
        {recommenders.map((name) => (
          <Radar
            key={name}
            name={name}
            dataKey={name}
            stroke={COLORS[name as keyof typeof COLORS] ?? "#888"}
            fill={COLORS[name as keyof typeof COLORS] ?? "#888"}
            fillOpacity={0.15}
          />
        ))}
        <Legend wrapperStyle={{ fontSize: 12, color: "#a1a1aa" }} />
        <Tooltip
          contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 8 }}
          labelStyle={{ color: "#fff" }}
          itemStyle={{ color: "#a1a1aa" }}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}

export function PrecisionRecallChart({ metrics }: Props) {
  const ks = [5, 10, 20];
  const recommenders = Object.keys(metrics);

  const precisionData = ks.map((k) => {
    const point: Record<string, string | number> = { k: `@${k}` };
    recommenders.forEach((name) => {
      point[name] = parseFloat(
        ((metrics[name][`precision@${k}` as keyof (typeof metrics)[string]] as number) ?? 0).toFixed(4),
      );
    });
    return point;
  });

  const recallData = ks.map((k) => {
    const point: Record<string, string | number> = { k: `@${k}` };
    recommenders.forEach((name) => {
      point[name] = parseFloat(
        ((metrics[name][`recall@${k}` as keyof (typeof metrics)[string]] as number) ?? 0).toFixed(4),
      );
    });
    return point;
  });

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div>
        <h3 className="mb-3 text-sm font-medium text-zinc-300">Precision@K</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={precisionData}>
            <CartesianGrid stroke="#3f3f46" strokeDasharray="3 3" />
            <XAxis dataKey="k" tick={{ fill: "#a1a1aa", fontSize: 12 }} />
            <YAxis tick={{ fill: "#a1a1aa", fontSize: 11 }} domain={[0, 1]} />
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 8 }}
              labelStyle={{ color: "#fff" }}
            />
            <Legend wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }} />
            {recommenders.map((name) => (
              <Bar key={name} dataKey={name} fill={COLORS[name as keyof typeof COLORS] ?? "#888"} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div>
        <h3 className="mb-3 text-sm font-medium text-zinc-300">Recall@K</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={recallData}>
            <CartesianGrid stroke="#3f3f46" strokeDasharray="3 3" />
            <XAxis dataKey="k" tick={{ fill: "#a1a1aa", fontSize: 12 }} />
            <YAxis tick={{ fill: "#a1a1aa", fontSize: 11 }} domain={[0, 1]} />
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 8 }}
              labelStyle={{ color: "#fff" }}
            />
            <Legend wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }} />
            {recommenders.map((name) => (
              <Bar key={name} dataKey={name} fill={COLORS[name as keyof typeof COLORS] ?? "#888"} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
