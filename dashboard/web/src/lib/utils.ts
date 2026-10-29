import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { EventType, SignalState } from "./types";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export const COLORS = {
  violation: "#ef4444",
  caution: "#f59e0b",
  legal: "#22c55e",
  primary: "#6366f1",
  slate: "#64748b",
  speed: "#c04ad0",
} as const;

export const EVENT_LABEL: Record<EventType, string> = {
  red_light_violation: "Red-light violation",
  amber_crossing: "Amber crossing",
  legal_crossing: "Legal crossing",
  overspeed: "Overspeed",
};

export const EVENT_COLOR: Record<EventType, string> = {
  red_light_violation: COLORS.violation,
  amber_crossing: COLORS.caution,
  legal_crossing: COLORS.legal,
  overspeed: COLORS.speed,
};

export const SIGNAL_COLOR: Record<SignalState | string, string> = {
  red: COLORS.violation,
  yellow: COLORS.caution,
  green: COLORS.legal,
  unknown: COLORS.slate,
};

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function fmtInt(n: number | null | undefined): string {
  return n == null ? "-" : Math.round(n).toLocaleString();
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "-";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString([], { dateStyle: "medium", timeStyle: "medium" });
}

export function fmtShortTime(iso: string, bucket: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  if (bucket === "day") return d.toLocaleDateString([], { month: "short", day: "numeric" });
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function pct(v: number | null | undefined): string {
  return v == null ? "n/a" : `${Math.round(v * 100)}%`;
}
