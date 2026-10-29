import { AlertOctagon, CheckCircle2, Gauge, TriangleAlert } from "lucide-react";
import type { EventType } from "@/lib/types";
import { EVENT_LABEL } from "@/lib/utils";
import { Badge } from "./ui/badge";

const MAP = {
  red_light_violation: { variant: "violation", Icon: AlertOctagon },
  amber_crossing: { variant: "caution", Icon: TriangleAlert },
  legal_crossing: { variant: "legal", Icon: CheckCircle2 },
  overspeed: { variant: "speed", Icon: Gauge },
} as const;

export function EventBadge({ type }: { type: EventType }) {
  const { variant, Icon } = MAP[type] ?? MAP.legal_crossing;
  return (
    <Badge variant={variant}>
      <Icon className="h-3 w-3" aria-hidden />
      {EVENT_LABEL[type] ?? type}
    </Badge>
  );
}
