import { useReducedMotion } from "framer-motion";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { NamedValue } from "@/lib/types";
import { SIGNAL_COLOR, titleCase } from "@/lib/utils";
import { ChartCard } from "./ChartCard";

interface Props { data?: NamedValue[] | null; loading: boolean }

/** How many crossings happened on each signal state. */
export function SignalBar({ data, loading }: Props) {
  const reduce = useReducedMotion();
  const empty = !data || data.every((d) => d.value === 0);
  return (
    <ChartCard title="Signal state at crossing" description="All crossings" loading={loading} empty={empty} height={280}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={(data ?? []).map((d) => ({ ...d, label: titleCase(d.name) }))} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" tickLine={false} axisLine={false} />
          <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
          <Tooltip cursor={{ fill: "hsl(var(--muted) / 0.5)" }} />
          <Bar dataKey="value" name="Crossings" radius={[8, 8, 0, 0]} isAnimationActive={!reduce} animationDuration={900}>
            {(data ?? []).map((d) => <Cell key={d.name} fill={SIGNAL_COLOR[d.name]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
