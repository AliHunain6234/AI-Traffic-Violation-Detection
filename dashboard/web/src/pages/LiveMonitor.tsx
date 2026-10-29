import { motion } from "framer-motion";
import { AlertTriangle, Play, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { PageTransition } from "@/components/PageTransition";
import { SignalLight } from "@/components/SignalLight";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Select } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { useCountUp } from "@/hooks/useCountUp";
import { useFetch } from "@/hooks/useEvents";
import { useLive } from "@/hooks/useLiveSocket";
import { api, streamUrl } from "@/lib/api";
import { cn, titleCase } from "@/lib/utils";

function Counter({ label, value, tone }: { label: string; value: number | undefined; tone?: string }) {
  const shown = useCountUp(value ?? 0, 500);
  return (
    <motion.div whileHover={{ y: -3 }} className="rounded-lg border border-border bg-card p-4 transition-shadow hover:shadow-lift">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      {value === undefined ? <Skeleton className="mt-2 h-8 w-16" /> : <p className={cn("text-3xl font-bold tabular-nums", tone)}>{Math.round(shown).toLocaleString()}</p>}
    </motion.div>
  );
}

/** Re-read the video list whenever a run starts or ends (new files may have been added). */
const running0 = (state?: string) => (state === "running" || state === "starting" ? 1 : 0);

const STATE_BADGE = { idle: "default", starting: "primary", running: "primary", finished: "legal", error: "violation" } as const;

export default function LiveMonitor() {
  const { tick, connected } = useLive();
  const [video, setVideo] = useState("");
  const [camera, setCamera] = useState("");
  const [tracker, setTracker] = useState("bytetrack");
  const [assume, setAssume] = useState("");
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState("");
  const [imgOk, setImgOk] = useState(true);

  const choices = useFetch((s) => api.videos(s), [], running0(tick?.run?.state));
  const run = tick?.run;
  const stats = tick?.stats;
  const state = run?.state ?? "idle";
  const running = state === "running" || state === "starting";

  // While a run is active the page drops its decorative effects (see [data-perf-auto] in index.css): the
  // detector and the browser share one CPU, and every animated layer is time taken from the detector.
  useEffect(() => {
    const root = document.documentElement;
    if (running) root.dataset.perfAuto = "lite"; else delete root.dataset.perfAuto;
    return () => { delete root.dataset.perfAuto; };
  }, [running]);

  const start = async () => {
    setBusy(true); setFormError("");
    try { await api.start({ video: video.trim(), tracker, ...(camera ? { camera } : {}), ...(assume ? { assume_signal: assume } : {}) }); }
    catch (e) { setFormError(e instanceof Error ? e.message : "Could not start"); }
    finally { setBusy(false); }
  };
  const stop = async () => { setBusy(true); try { await api.stop(); } finally { setBusy(false); } };

  return (
    <PageTransition>
      <div className="space-y-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Live monitor</h1>
            <p className="text-sm text-muted-foreground">Annotated stream with signal state and live counters</p>
          </div>
          <Badge variant={STATE_BADGE[state]}>{titleCase(state)}{stats?.video_timestamp && running ? ` · ${stats.video_timestamp}` : ""}</Badge>
        </div>

        <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
          <Card className="overflow-hidden">
            <div className="relative aspect-video w-full mjpeg">
              {imgOk ? (
                <img src={streamUrl()} alt="Annotated live video stream" className="h-full w-full object-contain" onError={() => setImgOk(false)} />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center text-sm text-muted-foreground">
                  <AlertTriangle className="h-6 w-6" />Stream unavailable.
                  <Button size="sm" variant="outline" onClick={() => setImgOk(true)}>Retry</Button>
                </div>
              )}
              {running && <span className="absolute left-3 top-3 flex items-center gap-1.5 rounded-full bg-black/60 px-2.5 py-1 text-xs font-semibold text-white"><span className="h-2 w-2 animate-pulseLight rounded-full bg-violation" />LIVE</span>}
            </div>
          </Card>
          <Card className="flex flex-row flex-wrap items-start justify-center gap-4 p-4 lg:flex-col lg:items-center lg:px-8">
            {Object.keys(stats?.signals ?? {}).length > 1 ? (
              Object.entries(stats?.signals ?? {}).map(([name, st]) => (
                <div key={name} className="flex flex-col items-center gap-1.5">
                  <SignalLight state={st} size="sm" />
                  <p className="max-w-[6rem] truncate text-center text-xs font-medium" title={name}>{name}: {titleCase(st)}</p>
                </div>
              ))
            ) : (
              <>
                <SignalLight state={stats?.signal_state} />
                <p className="text-center text-sm font-medium">{stats?.signal_state ? titleCase(stats.signal_state) : "No signal"}</p>
              </>
            )}
          </Card>
        </div>

        {running && stats?.timings_ms && (
          <p className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-border bg-card px-4 py-2 text-xs text-muted-foreground" aria-label="Performance">
            <span>Camera <b className="text-foreground">{stats.camera_fps?.toFixed(0) ?? "?"}</b> fps</span>
            <span>Detection <b className="text-foreground">{stats.fps?.toFixed(1)}</b> /s</span>
            <span>Per frame <b className="text-foreground">{stats.timings_ms.busy_ms ?? "?"}</b> ms (model {stats.timings_ms.detect_ms ?? "?"} ms)</span>
            {stats.cpu_percent != null && <span>CPU <b className={cn(stats.cpu_percent > 90 ? "text-caution" : "text-foreground")}>{Math.round(stats.cpu_percent)}%</b></span>}
            {(stats.cpu_percent ?? 0) > 90 && <span className="text-caution">The computer is at its limit: close other programs (screen recorder, extra browser tabs).</span>}
          </p>
        )}

        <section className="grid grid-cols-2 gap-4 md:grid-cols-5" aria-label="Live counters">
          <Counter label="Vehicles tracked" value={stats && (stats.vehicles_tracked ?? 0)} />
          <Counter label="Crossings" value={stats && (stats.crossings ?? 0)} />
          <Counter label="Red-light" value={stats && (stats.red_light_violations ?? 0)} tone="text-violation" />
          <Counter label="Amber" value={stats && (stats.amber_crossings ?? 0)} tone="text-caution" />
          <Counter label="Overspeed" value={stats && (stats.overspeed_events ?? 0)} tone="text-fuchsia-400" />
        </section>

        <Card>
          <CardHeader>
            <CardTitle>Process a video or live camera</CardTitle>
            <CardDescription>
              A video file on the server (inside an allowed folder), a camera URL (rtsp://… or http://…/video), or <code>0</code> for the server's webcam.
              {!connected && " (Live socket offline.)"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid gap-3 md:grid-cols-[1fr_9rem_9rem_11rem_auto]">
              <Input aria-label="Video path or camera URL" list="video-choices" placeholder="Pick a video, or type rtsp://…  |  http://…  |  0 (webcam)" value={video} onChange={(e) => setVideo(e.target.value)} disabled={running} />
              <datalist id="video-choices">{(choices.data?.videos ?? []).map((v) => <option key={v} value={v} />)}</datalist>
              <Select aria-label="Camera setup (stop line and signal)" title="Which camera setup (stop lines, signals, speed calibration) to use" value={camera || choices.data?.active_camera || ""} onChange={(e) => setCamera(e.target.value)} disabled={running}>
                {(choices.data?.cameras ?? ["default"]).map((c) => <option key={c} value={c}>{c}</option>)}
              </Select>
              <Select aria-label="Tracker" value={tracker} onChange={(e) => setTracker(e.target.value)} disabled={running}>
                <option value="botsort">BoT-SORT</option><option value="bytetrack">ByteTrack</option>
              </Select>
              <Select aria-label="Assume signal (testing only)" value={assume} onChange={(e) => setAssume(e.target.value)} disabled={running}>
                <option value="">Read real signal</option><option value="red">Test: always red</option><option value="yellow">Test: always yellow</option><option value="green">Test: always green</option>
              </Select>
              {running ? (
                <Button variant="danger" onClick={stop} disabled={busy}><Square className="h-4 w-4" />Stop</Button>
              ) : (
                <Button onClick={start} disabled={busy || !video.trim()}><Play className="h-4 w-4" />Start</Button>
              )}
            </div>
            {(formError || run?.error) && <p role="alert" className="text-sm text-violation">{formError || run?.error}</p>}
            {state === "finished" && run?.summary && <p className="text-sm text-muted-foreground">Finished: {String(run.summary.frames_processed)} frames at {Number(run.summary.avg_fps).toFixed(1)} FPS.</p>}
          </CardContent>
        </Card>
      </div>
    </PageTransition>
  );
}
