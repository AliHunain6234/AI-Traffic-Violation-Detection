import { useReducedMotion } from "framer-motion";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TimePoint } from "@/lib/types";
import { COLORS, fmtShortTime } from "@/lib/utils";
import { ChartCard } from "./ChartCard";

interface Props { data?: { bucket: string; points: TimePoint[] } | null; loading: boolean }

/** Violations over time: stacked areas that draw in from the left. */
export function ViolationsArea({ data, loading }: Props) {
  const reduce = useReducedMotion();
  const bucket = data?.bucket ?? "hour";
  return (
    <ChartCard title="Violations over time" description={`Per ${bucket}`} loading={loading} empty={!data?.points.length} height={280}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data?.points ?? []} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
          <defs>
            <linearGradient id="gRed" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={COLORS.violation} stopOpacity={0.6} /><stop offset="100%" stopColor={COLORS.violation} stopOpacity={0.02} /></linearGradient>
            <linearGradient id="gAmber" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={COLORS.caution} stopOpacity={0.55} /><stop offset="100%" stopColor={COLORS.caution} stopOpacity={0.02} /></linearGradient>
            <linearGradient id="gSpeed" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={COLORS.speed} stopOpacity={0.55} /><stop offset="100%" stopColor={COLORS.speed} stopOpacity={0.02} /></linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="t" tickFormatter={(v: string) => fmtShortTime(v, bucket)} minTickGap={24} tickLine={false} axisLine={false} />
          <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
          <Tooltip labelFormatter={(v) => new Date(String(v)).toLocaleString()} />
          <Area type="monotone" dataKey="red_light_violation" name="Red-light" stackId="1" stroke={COLORS.violation} strokeWidth={2} fill="url(#gRed)" isAnimationActive={!reduce} animationDuration={1200} />
          <Area type="monotone" dataKey="amber_crossing" name="Amber" stackId="1" stroke={COLORS.caution} strokeWidth={2} fill="url(#gAmber)" isAnimationActive={!reduce} animationDuration={1200} />
          <Area type="monotone" dataKey="overspeed" name="Overspeed" stackId="1" stroke={COLORS.speed} strokeWidth={2} fill="url(#gSpeed)" isAnimationActive={!reduce} animationDuration={1200} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
