"use client";

import { useState, useCallback, useTransition } from "react";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { BookCard } from "./BookCard";
import { api } from "@/lib/api";
import { ALGORITHMS } from "@/lib/constants";
import type { Algorithm, RecommendationItem, User } from "@/lib/types";

interface Props {
  users: User[];
  initialRecs: RecommendationItem[];
}

export function DemoClient({ users, initialRecs }: Props) {
  const [userId, setUserId] = useState<number | string>(users[0]?.user_id ?? 1);
  const [algorithm, setAlgorithm] = useState<Algorithm>("hybrid");
  const [cfWeight, setCfWeight] = useState(60);
  const [recs, setRecs] = useState<RecommendationItem[]>(initialRecs);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const fetch = useCallback(
    (uid: number | string, alg: Algorithm, weight: number) => {
      startTransition(async () => {
        setError(null);
        try {
          const res = await api.recommendations.get(
            uid,
            alg,
            10,
            alg === "hybrid" ? weight / 100 : undefined,
          );
          setRecs(res.recommendations);
        } catch (e) {
          setError((e as Error).message);
        }
      });
    },
    [],
  );

  function handleUserChange(val: string | null) {
    if (!val) return;
    const uid = isNaN(Number(val)) ? val : Number(val);
    setUserId(uid);
    fetch(uid, algorithm, cfWeight);
  }

  function handleAlgorithmChange(val: string | null) {
    if (!val) return;
    setAlgorithm(val as Algorithm);
    fetch(userId, val as Algorithm, cfWeight);
  }

  function handleWeightChange(val: number | readonly number[]) {
    const v = Array.isArray(val) ? val[0] : (val as number);
    setCfWeight(v);
    fetch(userId, algorithm, v);
  }

  const selectedAlg = ALGORITHMS.find((a) => a.id === algorithm)!;
  const selectedUser = users.find((u) => String(u.user_id) === String(userId));

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6 space-y-6">
        <div className="grid gap-4 sm:grid-cols-2">
          {/* User picker */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-zinc-300">User</label>
            <Select value={String(userId)} onValueChange={handleUserChange}>
              <SelectTrigger className="border-zinc-700 bg-zinc-800">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="border-zinc-700 bg-zinc-900">
                {users.map((u) => (
                  <SelectItem key={String(u.user_id)} value={String(u.user_id)}>
                    {u.display_name}
                    <span className="ml-2 text-xs text-zinc-500">
                      ({u.rating_count} ratings · {u.profile_type})
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Algorithm picker */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-zinc-300">Algorithm</label>
            <Select value={algorithm} onValueChange={(v) => handleAlgorithmChange(v as Algorithm)}>
              <SelectTrigger className="border-zinc-700 bg-zinc-800">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="border-zinc-700 bg-zinc-900">
                {ALGORITHMS.map((a) => (
                  <SelectItem key={a.id} value={a.id}>
                    <span className={a.color}>{a.shortLabel}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Hybrid weight slider */}
        {algorithm === "hybrid" && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-zinc-400">
                CF weight: <span className="font-mono text-white">{cfWeight}%</span>
              </span>
              <span className="text-zinc-400">
                CB weight: <span className="font-mono text-white">{100 - cfWeight}%</span>
              </span>
            </div>
            <Slider
              value={[cfWeight]}
              onValueChange={handleWeightChange}
              min={0}
              max={100}
              step={5}
              className="cursor-pointer"
            />
            <p className="text-xs text-zinc-500">
              Drag to override the adaptive CF/CB weighting
            </p>
          </div>
        )}

        {/* Algorithm explainer */}
        <div className={`rounded-lg border border-zinc-800 bg-zinc-950 p-4`}>
          <p className={`text-sm font-medium ${selectedAlg.color}`}>{selectedAlg.shortLabel}</p>
          <p className="mt-1 text-sm text-zinc-400">{selectedAlg.detail}</p>
          {selectedUser && (
            <div className="mt-3 flex gap-2 flex-wrap">
              <Badge variant="outline" className="border-zinc-700 text-zinc-400 text-xs">
                {selectedUser.rating_count} ratings
              </Badge>
              <Badge variant="outline" className="border-zinc-700 text-zinc-400 text-xs">
                avg {selectedUser.avg_rating.toFixed(1)} ★
              </Badge>
              <Badge variant="outline" className="border-zinc-700 text-zinc-400 text-xs">
                {selectedUser.profile_type}
              </Badge>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      <div>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">
            Recommendations
            {isPending && (
              <span className="ml-2 text-sm font-normal text-zinc-500 animate-pulse">
                loading…
              </span>
            )}
          </h2>
          <span className="text-sm text-zinc-500">{recs.length} results</span>
        </div>

        {error && (
          <div className="rounded-lg border border-red-900 bg-red-950/30 p-4 text-sm text-red-400">
            {error}
          </div>
        )}

        <div className="space-y-2">
          {recs.map((item, i) => (
            <BookCard key={item.item_id} item={item} index={i} />
          ))}
        </div>
      </div>
    </div>
  );
}
