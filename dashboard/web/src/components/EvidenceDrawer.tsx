import { AnimatePresence, motion, type PanInfo } from "framer-motion";
import { ChevronLeft, ChevronRight, ScanText } from "lucide-react";
import { useEffect, useState } from "react";
import { api, evidenceUrl } from "@/lib/api";
import { useFetch } from "@/hooks/useEvents";
import type { EventDetail } from "@/lib/types";
import { cn, fmtTime, pct, titleCase } from "@/lib/utils";
import { EventBadge } from "./EventBadge";
import { Dialog, DialogDescription, DialogTitle, DrawerContent } from "./ui/dialog";
import { Skeleton } from "./ui/skeleton";

const FRAMES = [
  { key: "before", label: "Before" },
  { key: "crossing", label: "At crossing" },
  { key: "after", label: "After" },
] as const;
const SWIPE_PX = 60;

function Carousel({ detail }: { detail: EventDetail }) {
  const slides = FRAMES.filter((f) => detail.evidence[f.key]);
  const [index, setIndex] = useState(Math.max(0, slides.findIndex((s) => s.key === "crossing")));
  const [dir, setDir] = useState(1);
  const go = (next: number) => {
    if (next < 0 || next >= slides.length) return;
    setDir(next > index ? 1 : -1);
    setIndex(next);
  };
  const onDragEnd = (_: unknown, info: PanInfo) => {
    if (info.offset.x < -SWIPE_PX) go(index + 1);
    else if (info.offset.x > SWIPE_PX) go(index - 1);
  };
  if (!slides.length) return <p className="rounded-lg border border-dashed border-border p-8 text-center text-sm text-muted-foreground">No frames saved for this event.</p>;
  const slide = slides[index];
  return (
    <div>
      <div className="relative aspect-video overflow-hidden rounded-lg border border-border bg-black">
        <AnimatePresence initial={false} custom={dir} mode="popLayout">
          <motion.img
            key={slide.key}
            src={evidenceUrl(detail.evidence[slide.key])}
            alt={`${slide.label} frame`}
            className="absolute inset-0 h-full w-full cursor-grab object-contain active:cursor-grabbing"
            custom={dir}
            variants={{ enter: (d: number) => ({ x: d * 80, opacity: 0 }), center: { x: 0, opacity: 1 }, exit: (d: number) => ({ x: d * -80, opacity: 0 }) }}
            initial="enter" animate="center" exit="exit"
            transition={{ duration: 0.25 }}
            drag="x" dragConstraints={{ left: 0, right: 0 }} dragElastic={0.4} onDragEnd={onDragEnd}
            draggable={false}
          />
        </AnimatePresence>
        {([[-1, ChevronLeft, "left-2", "Previous frame"], [1, ChevronRight, "right-2", "Next frame"]] as const).map(([d, Icon, pos, label]) => (
          <button
            key={label}
            aria-label={label}
            onClick={() => go(index + d)}
            disabled={index + d < 0 || index + d >= slides.length}
            className={cn("absolute top-1/2 -translate-y-1/2 rounded-full bg-black/60 p-1.5 text-white transition hover:bg-black/80 disabled:opacity-30", pos)}
          >
            <Icon className="h-5 w-5" />
          </button>
        ))}
      </div>
      <div className="mt-3 flex items-center justify-center gap-2">
        {slides.map((s, i) => (
          <button key={s.key} onClick={() => go(i)} aria-label={s.label} className="group flex flex-col items-center gap-1">
            <span className={cn("h-1.5 rounded-full transition-all", i === index ? "w-8 bg-primary" : "w-4 bg-border group-hover:bg-muted-foreground")} />
            <span className={cn("text-[11px]", i === index ? "text-foreground" : "text-muted-foreground")}>{s.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function Thumb({ src, label }: { src?: string; label: string }) {
  return (
    <figure className="space-y-1.5">
      <div className="flex aspect-[4/3] items-center justify-center overflow-hidden rounded-lg border border-border bg-muted">
        {src ? <img src={evidenceUrl(src)} alt={label} className="h-full w-full object-contain" /> : <span className="px-2 text-center text-xs text-muted-foreground">Not available</span>}
      </div>
      <figcaption className="text-xs text-muted-foreground">{label}</figcaption>
    </figure>
  );
}

function Body({ id }: { id: string }) {
  const { data, loading, error } = useFetch<EventDetail>((s) => api.event(id, s), [id]);
  if (error) return <p className="p-6 text-sm text-violation">Could not load event: {error}</p>;
  if (loading || !data) {
    return (
      <div className="space-y-4 p-6">
        <Skeleton className="h-6 w-2/3" /><Skeleton className="aspect-video w-full" />
        <div className="grid grid-cols-2 gap-3"><Skeleton className="h-32" /><Skeleton className="h-32" /></div>
      </div>
    );
  }
  const plate = data.metadata.plate;
  const rows: [string, string][] = [
    ["Event ID", data.event_id], ["Track", `#${data.track_id}`], ["Vehicle", titleCase(data.vehicle_class)],
    ["Signal at crossing", titleCase(data.signal_state)], ["Frame", String(data.frame_no)],
    ["Video time", data.video_timestamp], ["Wall clock", fmtTime(data.time)],
    ["Detection confidence", pct(data.confidence)], ["Camera", data.camera_id || "-"],
    ...(data.approach ? [["Stop line", data.approach]] as [string, string][] : []),
    ...(data.speed_kmh != null ? [["Measured speed", `${data.speed_kmh.toFixed(1)} km/h`], ["Speed limit", `${data.limit_kmh ?? "-"} km/h`]] as [string, string][] : []),
  ];
  return (
    <motion.div className="space-y-6 overflow-y-auto p-6" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <Carousel detail={data} />
      <div className="grid grid-cols-3 gap-3">
        <Thumb src={data.evidence.vehicle} label="Vehicle" />
        <Thumb src={data.evidence.plate} label="Plate" />
        <Thumb src={data.evidence.plate_enhanced} label="Plate (enhanced)" />
      </div>
      <div className="rounded-lg border border-border bg-muted/40 p-4">
        <p className="mb-1 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground"><ScanText className="h-4 w-4" />OCR result</p>
        {data.plate_text ? (
          <p className="font-mono text-2xl font-bold tracking-widest">{data.plate_text} <span className="ml-2 font-sans text-sm font-medium text-muted-foreground">{pct(data.plate_conf)} confidence</span></p>
        ) : (
          <p className="text-sm text-muted-foreground">{plate?.note || "No plate was read for this vehicle."}</p>
        )}
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="min-w-0"><dt className="text-xs text-muted-foreground">{k}</dt><dd className="truncate font-medium" title={v}>{v}</dd></div>
        ))}
      </dl>
    </motion.div>
  );
}

/** Slide-in drawer with the full evidence package of one event. */
export function EvidenceDrawer({ eventId, onClose }: { eventId: string | null; onClose: () => void }) {
  const [lastId, setLastId] = useState<string | null>(eventId);
  useEffect(() => { if (eventId) setLastId(eventId); }, [eventId]);
  const open = eventId !== null;
  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <AnimatePresence>
        {open && lastId && (
          <DrawerContent key="drawer" aria-describedby="drawer-desc">
            <header className="space-y-2 border-b border-border p-6 pr-14">
              <DialogTitle className="text-lg font-semibold">Evidence package</DialogTitle>
              <DialogDescription id="drawer-desc" className="text-sm text-muted-foreground">{lastId}</DialogDescription>
              <DrawerBadge id={lastId} />
            </header>
            <div className="flex-1 overflow-y-auto"><Body id={lastId} /></div>
          </DrawerContent>
        )}
      </AnimatePresence>
    </Dialog>
  );
}

function DrawerBadge({ id }: { id: string }) {
  const { data } = useFetch<EventDetail>((s) => api.event(id, s), [id]);
  return data ? <EventBadge type={data.event_type} /> : <Skeleton className="h-6 w-32" />;
}
