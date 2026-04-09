"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ALGORITHMS } from "@/lib/constants";

export function AlgorithmCards() {
  return (
    <section className="py-12">
      <h2 className="mb-8 text-center text-2xl font-semibold text-white">
        Three Algorithms, One System
      </h2>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {ALGORITHMS.map((alg, i) => (
          <motion.div
            key={alg.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1, duration: 0.4 }}
          >
            <Card className="h-full border-zinc-800 bg-zinc-900 transition-colors hover:border-zinc-700">
              <CardHeader className="pb-2">
                <CardTitle className={`text-base font-semibold ${alg.color}`}>
                  {alg.shortLabel}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-zinc-400">{alg.description}</p>
                <p className="mt-3 text-xs leading-relaxed text-zinc-500">{alg.detail}</p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
