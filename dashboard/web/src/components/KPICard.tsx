import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { useCountUp } from "@/hooks/useCountUp";
import { cn } from "@/lib/utils";
import { Card } from "./ui/card";
import { Skeleton } from "./ui/skeleton";

type Tone = "violation" | "caution" | "legal" | "primary" | "neutral" | "speed";

// icon: tinted tile; bar: accent strip along the top edge; glow: halo behind the tile.
const TONES: Record<Tone, { icon: string; bar: string; glow: string }> = {
  violation: { icon: "bg-violation/15 text-violation", bar: "from-violation/0 via-violation to-violation/0", glow: "bg-violation" },
  caution: { icon: "bg-caution/15 text-caution", bar: "from-caution/0 via-caution to-caution/0", glow: "bg-caution" },
  legal: { icon: "bg-legal/15 text-legal", bar: "from-legal/0 via-legal to-legal/0", glow: "bg-legal" },
  primary: { icon: "bg-primary/15 text-primary", bar: "from-primary/0 via-primary to-primary/0", glow: "bg-primary" },
  neutral: { icon: "bg-muted text-muted-foreground", bar: "from-foreground/0 via-foreground/40 to-foreground/0", glow: "bg-foreground" },
  speed: { icon: "bg-fuchsia-500/15 text-fuchsia-400", bar: "from-fuchsia-500/0 via-fuchsia-500 to-fuchsia-500/0", glow: "bg-fuchsia-500" },
};

interface Props {
  label: string;
  value: number | null | undefined;
  icon: LucideIcon;
  tone?: Tone;
  decimals?: number;
  suffix?: string;
  loading?: boolean;
  index?: number;
}

/** KPI tile: the number counts up on load and whenever it changes; the icon tilts and glows on hover. */
export function KPICard({ label, value, icon: Icon, tone = "neutral", decimals = 0, suffix = "", loading, index = 0 }: Props) {
  const shown = useCountUp(value ?? 0);
  const t = TONES[tone];
  return (
    <motion.div
      initial={{ opacity: 0, y: 24, scale: 0.94 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay: index * 0.06, type: "spring", stiffness: 260, damping: 22 }}
      whileHover={{ y: -6, scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <Card className="group relative flex items-center gap-4 p-5">
        <span className={cn("pointer-events-none absolute inset-x-6 top-0 h-px bg-gradient-to-r opacity-70 transition-opacity group-hover:opacity-100", t.bar)} />
        <div className="relative shrink-0">
          <span className={cn("absolute inset-0 rounded-xl opacity-0 blur-xl transition-opacity duration-300 group-hover:opacity-50", t.glow)} />
          <div className={cn("relative flex h-12 w-12 items-center justify-center rounded-xl transition-transform duration-300 group-hover:-rotate-6 group-hover:scale-110", t.icon)}>
            <Icon className="h-6 w-6" aria-hidden />
          </div>
        </div>
        <div className="min-w-0">
          <p className="truncate text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">{label}</p>
          {loading || value == null ? (
            <Skeleton className="mt-2 h-8 w-20" />
          ) : (
            <p className="text-3xl font-bold tabular-nums tracking-tight">
              {shown.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}
              {suffix && <span className="ml-1 text-sm font-medium text-muted-foreground">{suffix}</span>}
            </p>
          )}
        </div>
      </Card>
    </motion.div>
  );
}
