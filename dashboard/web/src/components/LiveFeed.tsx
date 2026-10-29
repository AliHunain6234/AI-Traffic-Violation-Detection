import { AnimatePresence, motion } from "framer-motion";
import { Radio } from "lucide-react";
import type { EventRow } from "@/lib/types";
import { cn, fmtTime } from "@/lib/utils";
import { EventBadge } from "./EventBadge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Skeleton } from "./ui/skeleton";

interface Props {
  events: EventRow[];
  connected: boolean;
  onSelect: (id: string) => void;
}

/** Newest events slide in from the top; red-light violations glow softly. */
export function LiveFeed({ events, connected, onSelect }: Props) {
  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-row items-center justify-between">
        <div>
          <CardTitle>Live feed</CardTitle>
          <CardDescription>New events as they are detected</CardDescription>
        </div>
        <span className={cn("flex items-center gap-1.5 text-xs", connected ? "text-legal" : "text-muted-foreground")}>
          <Radio className="h-3.5 w-3.5" aria-hidden />{connected ? "Live" : "Reconnecting"}
        </span>
      </CardHeader>
      <CardContent className="min-h-[18rem] flex-1 overflow-y-auto">
        {!connected && events.length === 0 ? (
          <div className="space-y-3">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-14 w-full" />)}</div>
        ) : events.length === 0 ? (
          <p className="py-10 text-center text-sm text-muted-foreground">Waiting for events. Start a video from the Live monitor.</p>
        ) : (
          <ul className="space-y-2">
            <AnimatePresence initial={false}>
              {events.map((e) => (
                <motion.li
                  key={e.event_id}
                  layout
                  initial={{ opacity: 0, y: -28, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ type: "spring", stiffness: 380, damping: 32 }}
                >
                  <button
                    onClick={() => onSelect(e.event_id)}
                    className={cn(
                      "flex w-full items-center justify-between gap-3 rounded-lg border border-border bg-muted/40 px-3 py-2.5 text-left transition hover:-translate-y-0.5 hover:bg-muted",
                      e.event_type === "red_light_violation" && "border-violation/40 shadow-glow",
                    )}
                  >
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">#{e.track_id} {e.vehicle_class}{e.plate_text ? ` · ${e.plate_text}` : ""}</p>
                      <p className="truncate text-xs text-muted-foreground">{fmtTime(e.time)} · frame {e.frame_no}</p>
                    </div>
                    <EventBadge type={e.event_type} />
                  </button>
                </motion.li>
              ))}
            </AnimatePresence>
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
