import { useReducedMotion } from "framer-motion";
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import type { NamedValue } from "@/lib/types";
import { COLORS, titleCase } from "@/lib/utils";
import { ChartCard } from "./ChartCard";

const PALETTE = [COLORS.violation, COLORS.primary, COLORS.caution, "#06b6d4", "#a855f7", COLORS.slate];

interface Props { data?: NamedValue[] | null; loading: boolean }

/** Violations by vehicle class. */
export function ClassDonut({ data, loading }: Props) {
  const reduce = useReducedMotion();
  const rows = (data ?? []).map((d) => ({ ...d, name: titleCase(d.name) }));
  return (
    <ChartCard title="Violations by vehicle class" description="Red-light and amber" loading={loading} empty={!rows.length} height={280}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={rows} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="82%" paddingAngle={3} stroke="none" isAnimationActive={!reduce} animationDuration={1000}>
            {rows.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
          </Pie>
          <Tooltip />
          <Legend verticalAlign="bottom" iconType="circle" />
        </PieChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
