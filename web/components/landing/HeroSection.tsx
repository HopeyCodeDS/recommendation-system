"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { buttonVariants } from "@/components/ui/button";

const rotating = ["Collaborative Filtering", "Content-Based", "Hybrid"];

export function HeroSection() {
  return (
    <section className="py-20 text-center">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="space-y-6"
      >
        <div className="inline-flex items-center gap-2 rounded-full border border-amber-400/30 bg-amber-400/10 px-4 py-1.5 text-sm text-amber-400">
          Portfolio Project · Machine Learning
        </div>

        <h1 className="text-5xl font-bold tracking-tight text-white lg:text-6xl">
          A book recommender
          <br />
          <span className="text-amber-400">that adapts to you</span>
        </h1>

        <p className="mx-auto max-w-2xl text-lg text-zinc-400">
          Three recommendation algorithms — Collaborative Filtering, Content-Based, and a Hybrid —
          evaluated on real Goodreads data. Interact with live predictions and see how each
          algorithm differs.
        </p>

        <div className="flex items-center justify-center gap-4 pt-4">
          <Link
            href="/demo"
            className={buttonVariants({ size: "lg", className: "bg-amber-400 text-zinc-950 hover:bg-amber-300" })}
          >
            Try the Demo
          </Link>
          <Link
            href="/metrics"
            className={buttonVariants({ size: "lg", variant: "outline", className: "border-zinc-700 text-zinc-300" })}
          >
            View Metrics
          </Link>
        </div>

        <div className="flex items-center justify-center gap-3 pt-6 text-sm text-zinc-500">
          {rotating.map((name) => (
            <span
              key={name}
              className="rounded border border-zinc-800 bg-zinc-900 px-2.5 py-1 font-mono text-xs"
            >
              {name}
            </span>
          ))}
        </div>
      </motion.div>
    </section>
  );
}
