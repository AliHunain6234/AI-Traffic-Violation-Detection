import { motion } from "framer-motion";
import type { SignalState } from "@/lib/types";
import { cn } from "@/lib/utils";

const LAMPS = [
  { state: "red", on: "#ef4444", glow: "0 0 28px 6px rgba(239,68,68,0.65)" },
  { state: "yellow", on: "#f59e0b", glow: "0 0 28px 6px rgba(245,158,11,0.65)" },
  { state: "green", on: "#22c55e", glow: "0 0 28px 6px rgba(34,197,94,0.65)" },
] as const;

interface Props {
  state: SignalState | string | undefined;
  size?: "sm" | "lg";
  className?: string;
}

/** Traffic-light housing: the active lamp glows and pulses; an unknown state dims all lamps. */
export function SignalLight({ state, size = "lg", className }: Props) {
  const lamp = size === "lg" ? "h-14 w-14" : "h-6 w-6";
  return (
    <div
      role="img"
      aria-label={`Signal state: ${state ?? "unknown"}`}
      className={cn("inline-flex flex-col items-center gap-3 rounded-2xl border border-border bg-black/70 p-3 shadow-inner", size === "sm" && "gap-1.5 rounded-lg p-1.5", className)}
    >
      {LAMPS.map((l) => {
        const active = state === l.state;
        return (
          <motion.span
            key={l.state}
            className={cn("block rounded-full", lamp, active && "animate-pulseLight")}
            animate={{ backgroundColor: active ? l.on : "#1f2937", boxShadow: active ? l.glow : "0 0 0 0 rgba(0,0,0,0)", opacity: active ? 1 : 0.55 }}
            transition={{ duration: 0.35 }}
          />
        );
      })}
    </div>
  );
}
