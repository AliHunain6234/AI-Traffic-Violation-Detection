import { useReducedMotion } from "framer-motion";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { BenchmarkRow } from "@/lib/types";
import { COLORS } from "@/lib/utils";
import { ChartCard } from "./ChartCard";

interface Props {
  rows?: BenchmarkRow[] | null;
  loading: boolean;
  metric: keyof BenchmarkRow;
  title: string;
  description: string;
  color: string;
}

/** One benchmark metric across model + tracker combinations. */
export function BenchmarkChart({ rows, loading, metric, title, description, color }: Props) {
  const reduce = useReducedMotion();
  const data = (rows ?? []).filter((r) => !r.error).map((r) => ({ name: `${r.model.replace(/\.pt$/, "")} + ${r.tracker}`, value: Number(r[metric] ?? 0) }));
  return (
    <ChartCard title={title} description={description} loading={loading} empty={!data.length} emptyText="Run benchmark.py to see results" height={240}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 11 }} />
          <YAxis tickLine={false} axisLine={false} />
          <Tooltip cursor={{ fill: "hsl(var(--muted) / 0.5)" }} />
          <Legend />
          <Bar dataKey="value" name={title} fill={color || COLORS.primary} radius={[8, 8, 0, 0]} isAnimationActive={!reduce} animationDuration={900} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
