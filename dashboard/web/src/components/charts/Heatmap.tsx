import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import type { HeatCell } from "@/lib/types";
import { ChartCard } from "./ChartCard";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const HOURS = Array.from({ length: 24 }, (_, h) => h);

interface Props { data?: HeatCell[] | null; loading: boolean }

/** Violations per weekday and hour. Cell opacity scales with the count. */
export function Heatmap({ data, loading }: Props) {
  const [hover, setHover] = useState<string>("");
  const { grid, max } = useMemo(() => {
    const g = new Map<string, number>();
    let m = 0;
    (data ?? []).forEach((c) => { g.set(`${c.day}-${c.hour}`, c.count); m = Math.max(m, c.count); });
    return { grid: g, max: m };
  }, [data]);

  return (
    <ChartCard title="Hourly violation heatmap" description={hover || "Weekday × hour of day (local time of the footage)"} loading={loading} empty={!data?.length} height={260}>
      <div className="h-full overflow-x-auto">
        <div className="grid min-w-[640px] gap-1" style={{ gridTemplateColumns: "2.5rem repeat(24, minmax(0, 1fr))" }} role="table" aria-label="Violations by weekday and hour">
          <div />
          {HOURS.map((h) => <div key={h} className="text-center text-[10px] text-muted-foreground">{h % 3 === 0 ? h : ""}</div>)}
          {DAYS.map((d, di) => (
            <div key={d} className="contents" role="row">
              <div className="flex items-center text-xs text-muted-foreground">{d}</div>
              {HOURS.map((h) => {
                const count = grid.get(`${di}-${h}`) ?? 0;
                return (
                  <motion.div
                    key={h}
                    role="cell"
                    aria-label={`${d} ${h}:00, ${count} violations`}
                    className="aspect-square min-h-[1.1rem] rounded-sm border border-border/50"
                    style={{ backgroundColor: count ? `hsl(var(--violation) / ${0.18 + 0.82 * (count / max)})` : "hsl(var(--muted) / 0.5)" }}
                    initial={{ opacity: 0, scale: 0.6 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: (di * 24 + h) * 0.003, duration: 0.25 }}
                    whileHover={{ scale: 1.25, zIndex: 2 }}
                    onMouseEnter={() => setHover(`${d} ${String(h).padStart(2, "0")}:00 · ${count} violation${count === 1 ? "" : "s"}`)}
                    onMouseLeave={() => setHover("")}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </ChartCard>
  );
}
