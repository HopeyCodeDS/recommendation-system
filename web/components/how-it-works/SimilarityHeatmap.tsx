"use client";

import type { SimilarityMatrix } from "@/lib/types";

interface Props {
  data: SimilarityMatrix;
}

function cell(value: number) {
  const clamped = Math.max(0, Math.min(1, value));
  const opacity = Math.round(clamped * 100);
  return {
    bg: `rgba(251,191,36,${clamped * 0.8 + 0.05})`,
    text: clamped > 0.5 ? "#1c1917" : "#d4d4d8",
    label: clamped.toFixed(2),
    opacity,
  };
}

export function SimilarityHeatmap({ data }: Props) {
  const { users, matrix } = data;

  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            <th className="w-16 text-zinc-500 pb-2" />
            {users.map((u) => (
              <th key={u} className="w-16 pb-2 font-mono text-zinc-400 text-center">
                U{u}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, ri) => (
            <tr key={users[ri]}>
              <td className="pr-3 font-mono text-zinc-400 text-right py-0.5">U{users[ri]}</td>
              {row.map((val, ci) => {
                const { bg, text, label } = cell(val);
                return (
                  <td
                    key={ci}
                    className="w-14 h-10 text-center font-mono font-semibold rounded transition-all"
                    style={{ backgroundColor: bg, color: text }}
                    title={`sim(U${users[ri]}, U${users[ci]}) = ${val}`}
                  >
                    {label}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-xs text-zinc-500">
        Cosine similarity between users based on their rating vectors. 1.0 = identical taste.
      </p>
    </div>
  );
}
