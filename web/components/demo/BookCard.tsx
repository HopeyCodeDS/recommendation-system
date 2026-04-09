"use client";

import { motion } from "framer-motion";
import Image from "next/image";
import { Badge } from "@/components/ui/badge";
import type { RecommendationItem } from "@/lib/types";

interface Props {
  item: RecommendationItem;
  index: number;
}

function StarRating({ value }: { value: number }) {
  const full = Math.floor(value);
  const partial = value - full;
  return (
    <div className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className="relative inline-block text-zinc-700">
          <span className="text-base">★</span>
          {i <= full && (
            <span className="absolute inset-0 text-amber-400 text-base">★</span>
          )}
          {i === full + 1 && partial > 0 && (
            <span
              className="absolute inset-0 overflow-hidden text-amber-400 text-base"
              style={{ width: `${partial * 100}%` }}
            >
              ★
            </span>
          )}
        </span>
      ))}
      <span className="ml-1 text-xs text-zinc-400">{value.toFixed(1)}</span>
    </div>
  );
}

export function BookCard({ item, index }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
      className="flex gap-4 rounded-lg border border-zinc-800 bg-zinc-900 p-4 transition-colors hover:border-zinc-700"
    >
      {/* Cover */}
      <div className="shrink-0">
        {item.image_url ? (
          <Image
            src={item.image_url}
            alt={item.title ?? "Book cover"}
            width={56}
            height={84}
            className="rounded object-cover"
            unoptimized
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        ) : (
          <div className="flex h-[84px] w-[56px] items-center justify-center rounded bg-zinc-800 text-xs font-bold text-zinc-500">
            {item.title?.slice(0, 2).toUpperCase() ?? "#"}
          </div>
        )}
      </div>

      {/* Info */}
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="font-medium leading-tight text-white line-clamp-2">
              {item.title ?? `Book #${item.item_id}`}
            </p>
            {item.authors && (
              <p className="text-xs text-zinc-500">{item.authors}</p>
            )}
          </div>
          <span className="shrink-0 text-xs font-mono text-zinc-600">#{item.rank}</span>
        </div>

        <StarRating value={item.predicted_rating} />

        <p className="text-xs text-zinc-500 line-clamp-1">{item.explanation}</p>
      </div>

      {/* Confidence badge */}
      {item.confidence != null && (
        <div className="shrink-0 self-start">
          <Badge
            variant="outline"
            className={
              item.confidence >= 0.7
                ? "border-emerald-800 text-emerald-400"
                : item.confidence >= 0.4
                  ? "border-amber-800 text-amber-400"
                  : "border-zinc-700 text-zinc-500"
            }
          >
            {Math.round(item.confidence * 100)}%
          </Badge>
        </div>
      )}
    </motion.div>
  );
}
