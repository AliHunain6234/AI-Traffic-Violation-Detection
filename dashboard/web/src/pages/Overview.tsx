import { AlertOctagon, Car, Cpu, Flag, Gauge, ScanLine, TriangleAlert, Zap } from "lucide-react";
import { useMemo, useState } from "react";
import { ClassDonut } from "@/components/charts/ClassDonut";
import { SignalBar } from "@/components/charts/SignalBar";
import { ViolationsArea } from "@/components/charts/ViolationsArea";
import { EvidenceDrawer } from "@/components/EvidenceDrawer";
import { KPICard } from "@/components/KPICard";
import { LiveFeed } from "@/components/LiveFeed";
import { PageTransition } from "@/components/PageTransition";
import { useEvents, useFetch } from "@/hooks/useEvents";
import { useLive } from "@/hooks/useLiveSocket";
import { api } from "@/lib/api";
import type { EventRow, Kpis } from "@/lib/types";

const NO_FILTERS = {};

export default function Overview() {
  const { kpis: liveKpis, recent, connected, eventVersion } = useLive();
  const [selected, setSelected] = useState<string | null>(null);

  const kpiQ = useFetch<Kpis>((s) => api.kpis(NO_FILTERS, s), [], eventVersion);
  const series = useFetch((s) => api.timeseries(NO_FILTERS, s), [], eventVersion);
  const byClass = useFetch((s) => api.byClass(NO_FILTERS, s), [], eventVersion);
  const bySignal = useFetch((s) => api.bySignal(NO_FILTERS, s), [], eventVersion);
  const seed = useEvents(NO_FILTERS, { limit: 8, sort: "time", order: "desc" }, eventVersion);

  const k = liveKpis ?? kpiQ.data;
  const loadingK = !k;
  const feed = useMemo<EventRow[]>(() => {
    const seen = new Set<string>();
    return [...recent, ...(seed.data?.items ?? [])].filter((e) => !seen.has(e.event_id) && !!seen.add(e.event_id)).slice(0, 12);
  }, [recent, seed.data]);

  return (
    <PageTransition>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Overview</h1>
          <p className="text-sm text-muted-foreground">Live status of the enforcement pipeline</p>
        </div>

        <section aria-label="Key figures" className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <KPICard index={0} label="Vehicles tracked" value={k?.vehicles_tracked} icon={Car} tone="primary" loading={loadingK} />
          <KPICard index={1} label="Crossings" value={k?.crossings} icon={Flag} tone="neutral" loading={loadingK} />
          <KPICard index={2} label="Red-light violations" value={k?.red_light_violations} icon={AlertOctagon} tone="violation" loading={loadingK} />
          <KPICard index={3} label="Amber crossings" value={k?.amber_crossings} icon={TriangleAlert} tone="caution" loading={loadingK} />
          <KPICard index={4} label="Overspeed events" value={k?.overspeed_events} icon={Gauge} tone="speed" loading={loadingK} />
          <KPICard index={5} label="Top speed" value={k?.max_speed_kmh} icon={Zap} tone="speed" suffix="km/h" decimals={0} loading={loadingK} />
          <KPICard index={6} label="Plates read" value={k?.plates_read} icon={ScanLine} tone="legal" loading={loadingK} />
          <KPICard index={7} label="Processing FPS" value={k?.fps} icon={Cpu} tone="primary" decimals={1} loading={loadingK} />
        </section>

        <section className="grid gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2"><ViolationsArea data={series.data} loading={series.loading && !series.data} /></div>
          <ClassDonut data={byClass.data} loading={byClass.loading && !byClass.data} />
        </section>

        <section className="grid gap-4 lg:grid-cols-3">
          <SignalBar data={bySignal.data} loading={bySignal.loading && !bySignal.data} />
          <div className="lg:col-span-2"><LiveFeed events={feed} connected={connected} onSelect={setSelected} /></div>
        </section>
      </div>
      <EvidenceDrawer eventId={selected} onClose={() => setSelected(null)} />
    </PageTransition>
  );
}
