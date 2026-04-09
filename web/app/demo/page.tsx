import { DemoClient } from "@/components/demo/DemoClient";
import { api } from "@/lib/api";

export const metadata = {
  title: "Live Demo — BookRec",
  description: "Try the recommender system interactively with different algorithms and users.",
};

export default async function DemoPage() {
  const [users, initialRecs] = await Promise.all([
    api.users.list().catch(() => []),
    api.recommendations
      .get(1, "hybrid", 10)
      .then((r) => r.recommendations)
      .catch(() => []),
  ]);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Interactive Demo</h1>
        <p className="mt-2 text-zinc-400">
          Select a user and algorithm to see live recommendations. Drag the weight slider to
          override the hybrid&apos;s adaptive CF/CB balance.
        </p>
      </div>
      <DemoClient users={users} initialRecs={initialRecs} />
    </div>
  );
}
