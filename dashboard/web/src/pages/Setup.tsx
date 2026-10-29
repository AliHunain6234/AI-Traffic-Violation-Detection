import { motion } from "framer-motion";
import { ArrowLeftRight, Check, ImageDown, RotateCcw, Trash2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent } from "react";
import { PageTransition } from "@/components/PageTransition";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { useFetch } from "@/hooks/useEvents";
import { api, setupApi, type SetupCamera, type SetupSave } from "@/lib/api";
import { applyHomography, directionArrow, solveHomography, type Pt } from "@/lib/geometry";
import { cn } from "@/lib/utils";

type Mode = "line" | "roi" | "speed-lines" | "speed-grid";

const MODES: { id: Mode; label: string; points: number; help: string }[] = [
  { id: "line", label: "Stop line", points: 2, help: "Click two points along the stop line, then drag the dots to adjust. The arrow must point the way vehicles travel to count; use Flip if it does not." },
  { id: "roi", label: "Signal", points: 0, help: "Drag a tight box around the traffic-light head. Pick a frame where the light is lit." },
  { id: "speed-lines", label: "Speed: two lines", points: 4, help: "Click line A (2 points) then line B (2 points) across the road, and enter the REAL distance between them." },
  { id: "speed-grid", label: "Speed: road grid", points: 4, help: "Mark the corners of a rectangle on the road of known size, in this order: near-left, near-right, far-right, far-left. A 1 m grid is drawn to check it." },
];
const NAME = /^[A-Za-z0-9_-]{1,40}$/;
const COLORS = { line: "#38bdf8", roi: "#f59e0b", speed: "#e879f9", saved: "#94a3b8" };

export default function Setup() {
  const choices = useFetch((s) => api.videos(s), []);
  const [source, setSource] = useState("");
  const [index, setIndex] = useState(0);
  const [camera, setCamera] = useState("");
  const [approach, setApproach] = useState("");
  const [mode, setMode] = useState<Mode>("line");
  const [frame, setFrame] = useState<{ url: string; width: number; height: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [saved, setSaved] = useState<SetupCamera | null>(null);

  const [pts, setPts] = useState<Record<Mode, Pt[]>>({ line: [], roi: [], "speed-lines": [], "speed-grid": [] });
  const [direction, setDirection] = useState<"positive" | "negative">("positive");
  const [roi, setRoi] = useState<[number, number, number, number] | null>(null);
  const [draft, setDraft] = useState<[number, number, number, number] | null>(null);
  const [distance, setDistance] = useState("");
  const [rectW, setRectW] = useState("3.5");
  const [rectL, setRectL] = useState("12");
  const [ratio, setRatio] = useState("");
  const drag = useRef<{ kind: "point"; i: number } | { kind: "roi"; x: number; y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const cameraName = camera || choices.data?.active_camera || "";
  const m = MODES.find((x) => x.id === mode)!;

  useEffect(() => () => { if (frame) URL.revokeObjectURL(frame.url); }, [frame]);
  useEffect(() => {
    if (!cameraName || !NAME.test(cameraName)) { setSaved(null); return; }
    const ctrl = new AbortController();
    setupApi.camera(cameraName, ctrl.signal).then(setSaved).catch(() => setSaved(null));
    return () => ctrl.abort();
  }, [cameraName]);

  const loadFrame = async (at = index) => {
    if (!source.trim()) { setMsg({ ok: false, text: "Pick a video or type a camera URL first." }); return; }
    setLoading(true); setMsg(null);
    try { setFrame(await setupApi.frame(source.trim(), at)); }
    catch (e) { setFrame(null); setMsg({ ok: false, text: e instanceof Error ? e.message : "Could not load a frame" }); }
    finally { setLoading(false); }
  };

  const toFrame = useCallback((e: { clientX: number; clientY: number }): Pt => {
    const r = svgRef.current!.getBoundingClientRect();
    return [Math.round(((e.clientX - r.left) / r.width) * frame!.width), Math.round(((e.clientY - r.top) / r.height) * frame!.height)];
  }, [frame]);

  const onDown = (e: RPointerEvent<SVGSVGElement>) => {
    if (!frame || drag.current) return;
    const p = toFrame(e);
    if (mode === "roi") {
      drag.current = { kind: "roi", x: p[0], y: p[1] };
      setDraft([p[0], p[1], 0, 0]);
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
    } else if (pts[mode].length < m.points) {
      setPts((s) => ({ ...s, [mode]: [...s[mode], p] }));
    }
  };
  const onMove = (e: RPointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    if (!d || !frame) return;
    const p = toFrame(e);
    if (d.kind === "point") setPts((s) => ({ ...s, [mode]: s[mode].map((q, i) => (i === d.i ? p : q)) }));
    else setDraft([Math.min(d.x, p[0]), Math.min(d.y, p[1]), Math.abs(p[0] - d.x), Math.abs(p[1] - d.y)]);
  };
  const onUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d?.kind === "roi" && draft && draft[2] >= 6 && draft[3] >= 6) setRoi(draft);
    setDraft(null);
  };
  const grab = (i: number) => (e: RPointerEvent<SVGCircleElement>) => {
    e.stopPropagation();
    drag.current = { kind: "point", i };
    svgRef.current!.setPointerCapture(e.pointerId);
  };

  const reset = () => { setPts((s) => ({ ...s, [mode]: [] })); if (mode === "roi") setRoi(null); setMsg(null); };

  const gridH = useMemo(() => {
    const p = pts["speed-grid"];
    const w = parseFloat(rectW), l = parseFloat(rectL);
    if (p.length !== 4 || !(w > 0) || !(l > 0)) return null;
    return solveHomography([[0, 0], [w, 0], [w, l], [0, l]], p);
  }, [pts, rectW, rectL]);

  const save = async () => {
    if (!frame) return;
    if (!NAME.test(cameraName)) { setMsg({ ok: false, text: "Camera name: letters, digits, - and _ only." }); return; }
    if (approach && !NAME.test(approach)) { setMsg({ ok: false, text: "Approach name: letters, digits, - and _ only." }); return; }
    const body: SetupSave = { camera: cameraName, ...(approach ? { approach } : {}), frame_size: [frame.width, frame.height] };
    if (mode === "line") { if (pts.line.length !== 2) return setMsg({ ok: false, text: "Click two points for the line." }); body.stop_line = pts.line; body.direction = direction; }
    if (mode === "roi") {
      if (!roi) return setMsg({ ok: false, text: "Drag a box around the signal first." });
      body.signal_roi = [...roi];
      if (ratio.trim()) body.min_pixel_ratio = parseFloat(ratio);
    }
    if (mode === "speed-lines") {
      if (pts["speed-lines"].length !== 4) return setMsg({ ok: false, text: "Click 4 points: line A (2) and line B (2)." });
      if (!(parseFloat(distance) > 0)) return setMsg({ ok: false, text: "Enter the real distance between the lines (metres)." });
      body.speed = { method: "lines", points: pts["speed-lines"], distance_m: parseFloat(distance) };
    }
    if (mode === "speed-grid") {
      if (pts["speed-grid"].length !== 4) return setMsg({ ok: false, text: "Click the 4 corners of the road rectangle." });
      if (!(parseFloat(rectW) > 0 && parseFloat(rectL) > 0)) return setMsg({ ok: false, text: "Enter the rectangle's real width and length (metres)." });
      body.speed = { method: "homography", points: pts["speed-grid"], rect_m: [parseFloat(rectW), parseFloat(rectL)] };
    }
    try {
      const res = await setupApi.save(body);
      setSaved(res);
      setMsg({ ok: true, text: `Saved: ${res.saved.join(", ")} for camera "${cameraName}"${approach ? ` / ${approach}` : ""}. The next run uses it.` });
    } catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : "Save failed" }); }
  };

  const clearSpeed = async () => {
    if (!frame) return;
    try { setSaved(await setupApi.save({ camera: cameraName, frame_size: [frame.width, frame.height], clear_speed: true })); setMsg({ ok: true, text: "Speed calibration removed." }); }
    catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : "Failed" }); }
  };
  const removeApproach = async (name: string) => {
    try { setSaved(await setupApi.removeApproach(cameraName, name)); setMsg({ ok: true, text: `Removed approach "${name}".` }); }
    catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : "Failed" }); }
  };

  const W = frame?.width ?? 1, H = frame?.height ?? 1;
  const unit = Math.max(W, H) / 700;               // draw sizes in "screen-ish" units whatever the frame resolution
  const line = pts.line;
  const arrow = line.length === 2 ? directionArrow(line[0], line[1], direction, 70 * unit) : null;
  const cur = pts[mode];

  return (
    <PageTransition>
      <div className="space-y-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Camera setup</h1>
          <p className="text-sm text-muted-foreground">Draw the stop line, signal and speed calibration on a real frame. It is saved to config.yaml and used by the next run.</p>
        </div>

        <div className="space-y-5">
          <Card className="overflow-hidden">
            <div className="relative flex w-full justify-center bg-black/60" style={{ minHeight: frame ? undefined : "min(60vw, 26rem)" }}>
              {loading && <Skeleton className="absolute inset-0 rounded-none" />}
              {!frame && !loading && (
                <div className="flex flex-col items-center justify-center gap-2 p-6 text-center text-sm text-muted-foreground">
                  <ImageDown className="h-7 w-7" />Choose a video or camera below and press <b>Load frame</b>.
                </div>
              )}
              {frame && (
                // Sized to the frame's own ratio (full width, capped at 75vh) so pointer maths stays exact for tall frames too.
                <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="block touch-none select-none"
                     style={{ cursor: "crosshair", width: `min(100%, ${(75 * W) / H}vh)`, aspectRatio: `${W} / ${H}` }}
                     onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} aria-label="Frame to draw on">
                  <image href={frame.url} width={W} height={H} />
                  {/* what is already configured */}
                  {saved?.approaches.map((a) => (
                    <g key={a.name} opacity={0.85}>
                      {a.stop_line && <line x1={a.stop_line[0][0]} y1={a.stop_line[0][1]} x2={a.stop_line[1][0]} y2={a.stop_line[1][1]} stroke={COLORS.saved} strokeWidth={3 * unit} strokeDasharray={`${10 * unit} ${6 * unit}`} />}
                      {a.stop_line && <text x={a.stop_line[0][0] + 6 * unit} y={a.stop_line[0][1] - 8 * unit} fill={COLORS.saved} fontSize={14 * unit}>{a.name}</text>}
                      {a.signal_roi && <rect x={a.signal_roi[0]} y={a.signal_roi[1]} width={a.signal_roi[2]} height={a.signal_roi[3]} fill="none" stroke={COLORS.saved} strokeWidth={2 * unit} strokeDasharray={`${6 * unit} ${4 * unit}`} />}
                    </g>
                  ))}
                  {saved?.speed?.lines && saved.speed.lines.map((l, i) => <line key={i} x1={l[0][0]} y1={l[0][1]} x2={l[1][0]} y2={l[1][1]} stroke={COLORS.saved} strokeWidth={2 * unit} strokeDasharray={`${8 * unit} ${5 * unit}`} />)}
                  {saved?.speed?.image_points && <polygon points={saved.speed.image_points.map((p) => p.join(",")).join(" ")} fill="none" stroke={COLORS.saved} strokeWidth={2 * unit} strokeDasharray={`${8 * unit} ${5 * unit}`} />}

                  {/* what is being drawn now */}
                  {mode === "line" && line.length === 2 && <line x1={line[0][0]} y1={line[0][1]} x2={line[1][0]} y2={line[1][1]} stroke={COLORS.line} strokeWidth={4 * unit} />}
                  {mode === "line" && arrow && <line x1={arrow[0][0]} y1={arrow[0][1]} x2={arrow[1][0]} y2={arrow[1][1]} stroke="#4ade80" strokeWidth={4 * unit} markerEnd="url(#arrowhead)" />}
                  <defs><marker id="arrowhead" markerWidth="8" markerHeight="8" refX="5" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#4ade80" /></marker></defs>
                  {mode === "roi" && (draft ?? roi) && <rect x={(draft ?? roi)![0]} y={(draft ?? roi)![1]} width={(draft ?? roi)![2]} height={(draft ?? roi)![3]} fill="rgba(245,158,11,0.15)" stroke={COLORS.roi} strokeWidth={3 * unit} />}
                  {mode === "speed-lines" && [0, 2].map((s) => cur.length >= s + 2 && <line key={s} x1={cur[s][0]} y1={cur[s][1]} x2={cur[s + 1][0]} y2={cur[s + 1][1]} stroke={s ? "#fb923c" : COLORS.speed} strokeWidth={4 * unit} />)}
                  {mode === "speed-grid" && cur.length === 4 && <polygon points={cur.map((p) => p.join(",")).join(" ")} fill="rgba(232,121,249,0.10)" stroke={COLORS.speed} strokeWidth={3 * unit} />}
                  {mode === "speed-grid" && gridH && (() => {
                    const w = parseFloat(rectW), l = parseFloat(rectL), step = Math.max(1, Math.ceil(Math.max(w, l) / 24));
                    const lines: JSX.Element[] = [];
                    for (let x = 0; x <= w + 1e-6; x += step) { const a = applyHomography(gridH, [x, 0]), b = applyHomography(gridH, [x, l]); lines.push(<line key={`x${x}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#4ade80" strokeWidth={1.5 * unit} opacity={0.8} />); }
                    for (let y = 0; y <= l + 1e-6; y += step) { const a = applyHomography(gridH, [0, y]), b = applyHomography(gridH, [w, y]); lines.push(<line key={`y${y}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#4ade80" strokeWidth={1.5 * unit} opacity={0.8} />); }
                    return <g>{lines}</g>;
                  })()}
                  {mode !== "roi" && cur.map((p, i) => (
                    <g key={i}>
                      <circle cx={p[0]} cy={p[1]} r={9 * unit} fill="#fff" stroke="#0f172a" strokeWidth={2 * unit} style={{ cursor: "grab" }} onPointerDown={grab(i)} />
                      <text x={p[0] + 12 * unit} y={p[1] - 10 * unit} fill="#fff" fontSize={15 * unit} stroke="#000" strokeWidth={0.6 * unit} paintOrder="stroke">{i + 1}</text>
                    </g>
                  ))}
                </svg>
              )}
            </div>
          </Card>

          {/* Controls sit below the frame and reflow by width: 1 column on phones up to 4 on wide screens. */}
          <div className="grid items-start gap-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 17rem), 1fr))" }}>
            <Card>
              <CardHeader><CardTitle>1 · Source</CardTitle><CardDescription>A video file, a camera URL (rtsp:// or http://) or 0 for the webcam.</CardDescription></CardHeader>
              <CardContent className="space-y-3">
                <Input aria-label="Video or camera" list="setup-sources" placeholder="Pick a video or type a URL" value={source} onChange={(e) => setSource(e.target.value)} />
                <datalist id="setup-sources">{(choices.data?.videos ?? []).map((v) => <option key={v} value={v} />)}</datalist>
                <div className="flex flex-wrap items-center gap-2">
                  <Input aria-label="Frame number" type="number" min={0} value={index} onChange={(e) => setIndex(Math.max(0, Number(e.target.value) || 0))} className="w-24" />
                  <Button onClick={() => loadFrame()} disabled={loading}><ImageDown className="h-4 w-4" />Load frame</Button>
                  <Button variant="outline" size="sm" onClick={() => { const n = index + 30; setIndex(n); loadFrame(n); }} disabled={loading || !source}>+30</Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>2 · Camera and approach</CardTitle><CardDescription>An approach is one stop line with its own signal. Leave it empty for a single-line camera.</CardDescription></CardHeader>
              <CardContent className="space-y-3">
                <Input aria-label="Camera name" list="setup-cameras" placeholder="camera name, e.g. junction_1" value={camera || cameraName} onChange={(e) => setCamera(e.target.value)} />
                <datalist id="setup-cameras">{(choices.data?.cameras ?? []).map((c) => <option key={c} value={c} />)}</datalist>
                <Input aria-label="Approach name" placeholder="approach (optional), e.g. left_lanes" value={approach} onChange={(e) => setApproach(e.target.value)} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>3 · Draw</CardTitle><CardDescription>{m.help}</CardDescription></CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-2" role="tablist" aria-label="What to draw">
                  {MODES.map((x) => (
                    <button key={x.id} role="tab" aria-selected={mode === x.id} onClick={() => { setMode(x.id); setMsg(null); }}
                      className={cn("min-h-11 rounded-md border px-2 py-2 text-sm font-medium transition", mode === x.id ? "border-primary bg-primary/15 text-foreground" : "border-border text-muted-foreground hover:bg-muted")}>
                      {x.label}
                    </button>
                  ))}
                </div>
                {mode === "line" && <Button variant="outline" size="sm" onClick={() => setDirection((d) => (d === "positive" ? "negative" : "positive"))}><ArrowLeftRight className="h-4 w-4" />Flip direction ({direction})</Button>}
                {mode === "roi" && <Input aria-label="Lamp sensitivity" placeholder="small lamp? min pixel ratio, e.g. 0.005 (optional)" value={ratio} onChange={(e) => setRatio(e.target.value)} />}
                {mode === "speed-lines" && <Input aria-label="Distance between the lines in metres" placeholder="real distance between line A and B (m)" value={distance} onChange={(e) => setDistance(e.target.value)} />}
                {mode === "speed-grid" && (
                  <div className="grid grid-cols-2 gap-2">
                    <Input aria-label="Rectangle width in metres" placeholder="width (m)" value={rectW} onChange={(e) => setRectW(e.target.value)} />
                    <Input aria-label="Rectangle length in metres" placeholder="length (m)" value={rectL} onChange={(e) => setRectL(e.target.value)} />
                  </div>
                )}
                <div className="flex flex-wrap gap-2">
                  <Button onClick={save} disabled={!frame}><Check className="h-4 w-4" />Save</Button>
                  <Button variant="outline" onClick={reset}><RotateCcw className="h-4 w-4" />Reset</Button>
                  {mode.startsWith("speed") && saved?.speed && <Button variant="ghost" onClick={clearSpeed}><Trash2 className="h-4 w-4" />Remove speed</Button>}
                </div>
                {msg && <motion.p initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} role={msg.ok ? "status" : "alert"} className={cn("text-sm", msg.ok ? "text-legal" : "text-violation")}>{msg.text}</motion.p>}
              </CardContent>
            </Card>

            {saved?.exists && (
              <Card>
                <CardHeader><CardTitle>Configured for “{saved.id}”</CardTitle></CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {saved.approaches.map((a) => (
                    <div key={a.name} className="flex items-center justify-between gap-2 rounded-md border border-border px-3 py-2">
                      <div className="min-w-0"><p className="truncate font-medium">{a.name}</p>
                        <p className="flex gap-1.5 pt-0.5"><Badge variant={a.stop_line ? "legal" : "default"}>line</Badge><Badge variant={a.signal_roi ? "legal" : "caution"}>{a.signal_roi ? "signal" : "no signal"}</Badge></p></div>
                      {!saved.uses_legacy_single_line && <Button variant="ghost" size="icon" aria-label={`Remove ${a.name}`} onClick={() => removeApproach(a.name)}><Trash2 className="h-4 w-4" /></Button>}
                    </div>
                  ))}
                  <p className="text-muted-foreground">Speed: {saved.speed ? (saved.speed.method === "lines" ? `two lines, ${saved.speed.distance_m} m apart` : "road grid") : "not calibrated"}</p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </PageTransition>
  );
}
