import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { MetricsData } from "@/lib/types";

const ROWS: { key: string; label: string; better: "lower" | "higher" }[] = [
  { key: "rmse", label: "RMSE", better: "lower" },
  { key: "mae", label: "MAE", better: "lower" },
  { key: "precision@5", label: "Precision@5", better: "higher" },
  { key: "precision@10", label: "Precision@10", better: "higher" },
  { key: "precision@20", label: "Precision@20", better: "higher" },
  { key: "recall@5", label: "Recall@5", better: "higher" },
  { key: "recall@10", label: "Recall@10", better: "higher" },
  { key: "recall@20", label: "Recall@20", better: "higher" },
];

interface Props {
  metrics: MetricsData;
}

export function MetricsTable({ metrics }: Props) {
  const recommenders = Object.keys(metrics);

  function best(key: string, better: "lower" | "higher"): string | null {
    const vals = recommenders.map((name) => ({
      name,
      val: (metrics[name] as Record<string, number>)[key] ?? null,
    }));
    const valid = vals.filter((v) => v.val !== null);
    if (valid.length === 0) return null;
    return (
      better === "lower"
        ? valid.reduce((a, b) => (a.val! < b.val! ? a : b))
        : valid.reduce((a, b) => (a.val! > b.val! ? a : b))
    ).name;
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-zinc-800">
      <Table>
        <TableHeader>
          <TableRow className="border-zinc-800 hover:bg-transparent">
            <TableHead className="text-zinc-400">Metric</TableHead>
            {recommenders.map((name) => (
              <TableHead key={name} className="text-zinc-400">
                {name}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {ROWS.map(({ key, label, better }) => {
            const winner = best(key, better);
            return (
              <TableRow key={key} className="border-zinc-800 hover:bg-zinc-900/50">
                <TableCell className="font-mono text-xs text-zinc-400">{label}</TableCell>
                {recommenders.map((name) => {
                  const val = (metrics[name] as Record<string, number>)[key];
                  const isWinner = name === winner;
                  return (
                    <TableCell
                      key={name}
                      className={
                        isWinner
                          ? "font-semibold text-amber-400"
                          : "text-zinc-300"
                      }
                    >
                      {val != null ? val.toFixed(4) : "—"}
                      {isWinner && (
                        <span className="ml-1 text-[10px] text-amber-500">★</span>
                      )}
                    </TableCell>
                  );
                })}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
