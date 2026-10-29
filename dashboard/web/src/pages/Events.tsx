import { AnimatePresence, motion } from "framer-motion";
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight, Search, X } from "lucide-react";
import { useState } from "react";
import { EventBadge } from "@/components/EventBadge";
import { EvidenceDrawer } from "@/components/EvidenceDrawer";
import { ExportButtons } from "@/components/ExportButtons";
import { PageTransition } from "@/components/PageTransition";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input, Select } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader } from "@/components/ui/table";
import { useEvents } from "@/hooks/useEvents";
import { useLive } from "@/hooks/useLiveSocket";
import type { Filters } from "@/lib/types";
import { EVENT_LABEL, SIGNAL_COLOR, fmtTime, pct, titleCase } from "@/lib/utils";

const PAGE_SIZE = 15;
const COLUMNS: { key: string; label: string; sortable: boolean }[] = [
  { key: "time", label: "Time", sortable: true },
  { key: "event_type", label: "Event", sortable: true },
  { key: "vehicle_class", label: "Vehicle", sortable: true },
  { key: "signal_state", label: "Signal", sortable: true },
  { key: "speed_kmh", label: "Speed", sortable: true },
  { key: "plate_text", label: "Plate", sortable: true },
  { key: "confidence", label: "Conf.", sortable: true },
  { key: "frame_no", label: "Frame", sortable: true },
];

export default function Events() {
  const { eventVersion } = useLive();
  const [filters, setFilters] = useState<Filters>({});
  const [sort, setSort] = useState("time");
  const [order, setOrder] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);

  const { data, loading, error } = useEvents(filters, { sort, order, limit: PAGE_SIZE, offset: page * PAGE_SIZE }, eventVersion);
  const set = (patch: Partial<Filters>) => { setFilters((f) => ({ ...f, ...patch })); setPage(0); };
  const toggleSort = (key: string) => {
    if (key === sort) setOrder((o) => (o === "asc" ? "desc" : "asc"));
    else { setSort(key); setOrder("desc"); }
    setPage(0);
  };
  const active = Object.values(filters).some(Boolean);
  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const facets = data?.facets;

  return (
    <PageTransition>
      <div className="space-y-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Events</h1>
            <p className="text-sm text-muted-foreground">Search, filter and review every detected crossing</p>
          </div>
          <ExportButtons filters={filters} />
        </div>

        <Card className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="relative sm:col-span-2">
            <Search className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-muted-foreground" aria-hidden />
            <Input className="pl-9" placeholder="Search ID, plate, vehicle…" aria-label="Search" value={filters.q ?? ""} onChange={(e) => set({ q: e.target.value })} />
          </div>
          <Select aria-label="Event type" value={filters.event_type ?? ""} onChange={(e) => set({ event_type: e.target.value })}>
            <option value="">All event types</option>
            {(facets?.event_type ?? []).map((t) => <option key={t} value={t}>{EVENT_LABEL[t as keyof typeof EVENT_LABEL] ?? t}</option>)}
          </Select>
          <Select aria-label="Vehicle class" value={filters.vehicle_class ?? ""} onChange={(e) => set({ vehicle_class: e.target.value })}>
            <option value="">All vehicles</option>
            {(facets?.vehicle_class ?? []).map((t) => <option key={t} value={t}>{titleCase(t)}</option>)}
          </Select>
          <Select aria-label="Signal state" value={filters.signal_state ?? ""} onChange={(e) => set({ signal_state: e.target.value })}>
            <option value="">All signal states</option>
            {(facets?.signal_state ?? []).map((t) => <option key={t} value={t}>{titleCase(t)}</option>)}
          </Select>
          <Input aria-label="Plate text" placeholder="Plate contains…" value={filters.plate ?? ""} onChange={(e) => set({ plate: e.target.value })} />
          <Input type="date" aria-label="From date" value={filters.date_from ?? ""} onChange={(e) => set({ date_from: e.target.value })} />
          <Input type="date" aria-label="To date" value={filters.date_to ?? ""} onChange={(e) => set({ date_to: e.target.value })} />
          {active && (
            <Button variant="ghost" size="sm" className="justify-self-start" onClick={() => { setFilters({}); setPage(0); }}>
              <X className="h-4 w-4" />Clear filters
            </Button>
          )}
        </Card>

        <Card className="overflow-hidden">
          <Table>
            <TableHeader>
              <tr>
                {COLUMNS.map((c) => (
                  <TableHead key={c.key} aria-sort={sort === c.key ? (order === "asc" ? "ascending" : "descending") : "none"}>
                    <button className="inline-flex items-center gap-1 uppercase tracking-wide hover:text-foreground" onClick={() => toggleSort(c.key)}>
                      {c.label}
                      {sort === c.key ? (order === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />) : <ArrowUpDown className="h-3 w-3 opacity-40" />}
                    </button>
                  </TableHead>
                ))}
              </tr>
            </TableHeader>
            <TableBody>
              {loading && !data ? (
                Array.from({ length: 8 }, (_, i) => (
                  <tr key={i} className="border-b border-border"><td colSpan={COLUMNS.length} className="px-3 py-2.5"><Skeleton className="h-8 w-full" /></td></tr>
                ))
              ) : (
                <AnimatePresence mode="popLayout" initial>
                  {(data?.items ?? []).map((e, i) => (
                    <motion.tr
                      key={e.event_id}
                      layout="position"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -16 }}
                      transition={{ duration: 0.22, delay: Math.min(i, 12) * 0.025 }}
                      className="cursor-pointer border-b border-border transition-colors hover:bg-muted/60 focus-within:bg-muted/60"
                      onClick={() => setSelected(e.event_id)}
                      tabIndex={0}
                      onKeyDown={(ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); setSelected(e.event_id); } }}
                      aria-label={`Open evidence for ${e.event_id}`}
                    >
                      <TableCell>{fmtTime(e.time)}</TableCell>
                      <TableCell><EventBadge type={e.event_type} /></TableCell>
                      <TableCell>{titleCase(e.vehicle_class)} <span className="text-muted-foreground">#{e.track_id}</span></TableCell>
                      <TableCell>
                        <span className="inline-flex items-center gap-2">
                          <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: SIGNAL_COLOR[e.signal_state] ?? SIGNAL_COLOR.unknown, boxShadow: `0 0 8px ${SIGNAL_COLOR[e.signal_state] ?? "transparent"}` }} aria-hidden />
                          {titleCase(e.signal_state)}
                        </span>
                      </TableCell>
                      <TableCell className="tabular-nums">
                        {e.speed_kmh != null ? (
                          <span className="font-semibold text-fuchsia-400">{Math.round(e.speed_kmh)}<span className="font-normal text-muted-foreground"> / {Math.round(e.limit_kmh ?? 0)} km/h</span></span>
                        ) : <span className="text-muted-foreground">-</span>}
                      </TableCell>
                      <TableCell className="font-mono">{e.plate_text || <span className="text-muted-foreground">-</span>}</TableCell>
                      <TableCell>{pct(e.confidence)}</TableCell>
                      <TableCell className="tabular-nums">{e.frame_no}</TableCell>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              )}
            </TableBody>
          </Table>
          {error && <p className="p-4 text-sm text-violation">{error}</p>}
          {!loading && !error && data?.items.length === 0 && (
            <p className="p-10 text-center text-sm text-muted-foreground">{active ? "No events match these filters." : "No events yet. Process a video to see results."}</p>
          )}
          <div className="flex items-center justify-between border-t border-border px-4 py-3 text-sm text-muted-foreground">
            <span>{total.toLocaleString()} event{total === 1 ? "" : "s"}</span>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="icon" aria-label="Previous page" disabled={page === 0} onClick={() => setPage((p) => p - 1)}><ChevronLeft className="h-4 w-4" /></Button>
              <span className="tabular-nums">Page {page + 1} / {pages}</span>
              <Button variant="outline" size="icon" aria-label="Next page" disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)}><ChevronRight className="h-4 w-4" /></Button>
            </div>
          </div>
        </Card>
      </div>
      <EvidenceDrawer eventId={selected} onClose={() => setSelected(null)} />
    </PageTransition>
  );
}
