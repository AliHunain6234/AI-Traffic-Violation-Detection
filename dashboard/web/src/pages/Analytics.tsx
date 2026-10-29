import { BenchmarkChart } from "@/components/charts/BenchmarkChart";
import { Heatmap } from "@/components/charts/Heatmap";
import { RadialGauge } from "@/components/charts/RadialGauge";
import { PageTransition } from "@/components/PageTransition";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useFetch } from "@/hooks/useEvents";
import { useLive } from "@/hooks/useLiveSocket";
import { api } from "@/lib/api";
import type { Metric } from "@/lib/types";
import { COLORS, titleCase } from "@/lib/utils";

const NO_FILTERS = {};
const fmt = (v: number | null | undefined, d = 1) => (v == null ? "-" : v.toFixed(d));

function Evaluation() {
  const { data, loading } = useFetch((s) => api.evaluation(s), []);
  if (loading && !data) return <div className="grid gap-4 sm:grid-cols-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-52" />)}</div>;
  const o = data?.overall;
  if (!o) {
    return (
      <Card><CardContent className="p-8 text-center text-sm text-muted-foreground">
        No evaluation yet. Fill <code>ground_truth/ground_truth.csv</code> and run <code>python evaluate.py</code>.
      </CardContent></Card>
    );
  }
  const per = Object.entries(data?.per_type ?? {}).filter((e): e is [string, Metric] => e[1] != null);
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Detection quality</CardTitle>
          <CardDescription>
            {data?.ground_truth_events} ground-truth events vs {data?.predicted_events} predictions, ±{data?.tolerance_frames} frames
            {data?.mean_abs_frame_error != null && ` · mean frame error ${data.mean_abs_frame_error}`}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 justify-items-center gap-6 py-4 sm:grid-cols-3">
          <RadialGauge label="Precision" value={o.precision} color={COLORS.primary} />
          <RadialGauge label="Recall" value={o.recall} color={COLORS.legal} />
          <RadialGauge label="F1 score" value={o.f1} color={COLORS.caution} />
        </CardContent>
      </Card>
      <Card>
        <Table>
          <TableHeader><tr><TableHead>Event type</TableHead><TableHead>TP</TableHead><TableHead>FP</TableHead><TableHead>FN</TableHead><TableHead>Precision</TableHead><TableHead>Recall</TableHead><TableHead>F1</TableHead></tr></TableHeader>
          <TableBody>
            {[["Overall", o] as const, ...per.map(([k, m]) => [titleCase(k), m] as const)].map(([name, m]) => (
              <tr key={name} className="border-b border-border">
                <TableCell className="font-medium">{name}</TableCell><TableCell>{m.tp}</TableCell><TableCell>{m.fp}</TableCell><TableCell>{m.fn}</TableCell>
                <TableCell>{fmt(m.precision == null ? null : m.precision * 100, 0)}%</TableCell><TableCell>{fmt(m.recall == null ? null : m.recall * 100, 0)}%</TableCell><TableCell>{fmt(m.f1 == null ? null : m.f1 * 100, 0)}%</TableCell>
              </tr>
            ))}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
}

function Benchmark() {
  const { data, loading } = useFetch((s) => api.benchmark(s), []);
  const rows = data?.results ?? [];
  const l = loading && !data;
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <BenchmarkChart rows={rows} loading={l} metric="fps" title="Pipeline FPS" description="Higher is faster" color={COLORS.primary} />
        <BenchmarkChart rows={rows} loading={l} metric="infer_ms" title="Detect + track time (ms/frame)" description="Lower is faster" color={COLORS.caution} />
        <BenchmarkChart rows={rows} loading={l} metric="unique_ids" title="Unique track IDs" description="Fewer IDs on the same video means steadier tracking" color={COLORS.slate} />
        <BenchmarkChart rows={rows} loading={l} metric="violations" title="Violations found" description="Red-light + amber" color={COLORS.violation} />
      </div>
      {rows.length > 0 && (
        <Card>
          <Table>
            <TableHeader><tr>{["Model", "Tracker", "FPS", "ms/frame", "Detections", "IDs", "Short tracks", "Mean track", "Violations"].map((h) => <TableHead key={h}>{h}</TableHead>)}</tr></TableHeader>
            <TableBody>
              {rows.map((r) => (
                <tr key={`${r.model}-${r.tracker}`} className="border-b border-border">
                  <TableCell className="font-medium">{r.model}</TableCell><TableCell>{r.tracker}</TableCell>
                  {r.error ? <TableCell colSpan={7} className="text-violation">{r.error}</TableCell> : (
                    <><TableCell>{fmt(r.fps)}</TableCell><TableCell>{fmt(r.infer_ms)}</TableCell><TableCell>{r.detections}</TableCell><TableCell>{r.unique_ids}</TableCell><TableCell>{r.short_tracks}</TableCell><TableCell>{fmt(r.mean_track_len)}</TableCell><TableCell>{r.violations}</TableCell></>
                  )}
                </tr>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}
    </div>
  );
}

export default function Analytics() {
  const { eventVersion } = useLive();
  const heat = useFetch((s) => api.heatmap(NO_FILTERS, s), [], eventVersion);
  return (
    <PageTransition>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Analytics</h1>
          <p className="text-sm text-muted-foreground">Patterns, model comparison and evaluation against ground truth</p>
        </div>
        <Tabs defaultValue="patterns">
          <TabsList aria-label="Analytics sections">
            <TabsTrigger value="patterns">Patterns</TabsTrigger>
            <TabsTrigger value="benchmark">Benchmark</TabsTrigger>
            <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
          </TabsList>
          <TabsContent value="patterns" className="mt-4"><Heatmap data={heat.data} loading={heat.loading && !heat.data} /></TabsContent>
          <TabsContent value="benchmark" className="mt-4"><Benchmark /></TabsContent>
          <TabsContent value="evaluation" className="mt-4"><Evaluation /></TabsContent>
        </Tabs>
      </div>
    </PageTransition>
  );
}
