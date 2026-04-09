"use client";

import { useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import type { Book, TFIDFTerm } from "@/lib/types";
import { api } from "@/lib/api";

interface Props {
  books: Pick<Book, "book_id" | "title">[];
  initialTerms: TFIDFTerm[];
  initialBookId: number;
}

export function TFIDFVisualizer({ books, initialTerms, initialBookId }: Props) {
  const [terms, setTerms] = useState<TFIDFTerm[]>(initialTerms);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState(String(initialBookId));

  async function handleChange(val: string | null) {
    if (!val) return;
    setSelectedId(val);
    setLoading(true);
    try {
      const t = await api.tfidf.terms(val);
      setTerms(t);
    } catch {
      setTerms([]);
    } finally {
      setLoading(false);
    }
  }

  const chartData = terms.map((t) => ({ term: t.term, score: parseFloat(t.score.toFixed(4)) }));

  return (
    <div className="space-y-4">
      <Select value={selectedId} onValueChange={handleChange}>
        <SelectTrigger className="w-64 border-zinc-700 bg-zinc-800">
          <SelectValue placeholder="Select a book…" />
        </SelectTrigger>
        <SelectContent className="border-zinc-700 bg-zinc-900 max-h-60">
          {books.map((b) => (
            <SelectItem key={b.book_id} value={String(b.book_id)}>
              {b.title}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {loading && <p className="text-sm text-zinc-500 animate-pulse">Loading…</p>}

      {!loading && chartData.length === 0 && (
        <p className="text-sm text-zinc-500">No TF-IDF data for this book (not in content model).</p>
      )}

      {!loading && chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 60, right: 20 }}>
            <XAxis type="number" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="term"
              tick={{ fill: "#d4d4d8", fontSize: 12, fontFamily: "monospace" }}
              width={60}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 8 }}
              itemStyle={{ color: "#fbbf24" }}
              cursor={{ fill: "rgba(255,255,255,0.04)" }}
            />
            <Bar dataKey="score" radius={[0, 4, 4, 0]}>
              {chartData.map((_, i) => (
                <Cell
                  key={i}
                  fill={`rgba(251,191,36,${0.4 + (0.6 * (chartData.length - i)) / chartData.length})`}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
      <p className="text-xs text-zinc-500">
        TF-IDF scores for top-10 terms per book. Higher = more distinctive for this book.
      </p>
    </div>
  );
}
