import { motion, useReducedMotion } from "framer-motion";
import { useCountUp } from "@/hooks/useCountUp";

interface Props { label: string; value: number | null | undefined; color: string }

const R = 52;
const C = 2 * Math.PI * R;

/** Radial gauge for a 0-1 metric (precision / recall / F1); the arc animates from empty. */
export function RadialGauge({ label, value, color }: Props) {
  const reduce = useReducedMotion();
  const v = value ?? 0;
  const shown = useCountUp(v * 100, 1000);
  return (
    <figure className="flex flex-col items-center gap-2" aria-label={`${label}: ${value == null ? "not available" : Math.round(v * 100) + "%"}`}>
      <svg viewBox="0 0 140 140" className="h-36 w-36 -rotate-90">
        <circle cx="70" cy="70" r={R} fill="none" stroke="hsl(var(--muted))" strokeWidth="12" />
        <motion.circle
          cx="70" cy="70" r={R} fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
          strokeDasharray={C}
          initial={{ strokeDashoffset: C }}
          animate={{ strokeDashoffset: C * (1 - v) }}
          transition={reduce ? { duration: 0 } : { duration: 1.1, ease: "easeOut" }}
        />
        <text x="70" y="70" textAnchor="middle" dominantBaseline="central" style={{ fill: "hsl(var(--foreground))", fontSize: 26, fontWeight: 700 }} transform="rotate(90 70 70)">
          {value == null ? "n/a" : `${Math.round(shown)}%`}
        </text>
      </svg>
      <figcaption className="text-sm font-medium text-muted-foreground">{label}</figcaption>
    </figure>
  );
}
