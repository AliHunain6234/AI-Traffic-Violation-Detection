import type {
  BenchmarkRow, Evaluation, EventDetail, EventList, Filters, HeatCell, Kpis, NamedValue, RunStatus, TimePoint,
} from "./types";

const TOKEN_KEY = "tvd_token";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export function getToken(): string {
  try { return localStorage.getItem(TOKEN_KEY) ?? ""; } catch { return ""; }
}
export function setToken(token: string): void {
  try { token ? localStorage.setItem(TOKEN_KEY, token) : localStorage.removeItem(TOKEN_KEY); } catch { /* storage blocked */ }
}

export function qs(params: Record<string, string | number | undefined | null>): string {
  const sp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  });
  const s = sp.toString();
  return s ? `?${s}` : "";
}

/** Add the token as a query parameter for URLs the browser loads itself (<img>, downloads). */
export function withToken(url: string): string {
  const token = getToken();
  if (!token) return url;
  return url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token);
}

async function request<T>(path: string, init: RequestInit = {}, signal?: AbortSignal): Promise<T> {
  const token = getToken();
  const res = await fetch(path, {
    ...init,
    signal,
    headers: {
      Accept: "application/json",
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
  if (res.status === 401) window.dispatchEvent(new Event("tvd-unauthorized"));
  if (!res.ok) {
    let detail = res.statusText;
    try { const body = await res.json(); detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch { /* not JSON */ }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export interface ListParams { sort?: string; order?: "asc" | "desc"; limit?: number; offset?: number }

export const api = {
  kpis: (f: Filters, s?: AbortSignal) => request<Kpis>(`/api/kpis${qs({ ...f })}`, {}, s),
  events: (f: Filters, p: ListParams, s?: AbortSignal) => request<EventList>(`/api/events${qs({ ...f, ...p })}`, {}, s),
  event: (id: string, s?: AbortSignal) => request<EventDetail>(`/api/events/${encodeURIComponent(id)}`, {}, s),
  timeseries: (f: Filters, s?: AbortSignal) => request<{ bucket: string; points: TimePoint[] }>(`/api/analytics/timeseries${qs({ ...f })}`, {}, s),
  byClass: (f: Filters, s?: AbortSignal) => request<NamedValue[]>(`/api/analytics/by-class${qs({ ...f })}`, {}, s),
  bySignal: (f: Filters, s?: AbortSignal) => request<NamedValue[]>(`/api/analytics/by-signal${qs({ ...f })}`, {}, s),
  heatmap: (f: Filters, s?: AbortSignal) => request<HeatCell[]>(`/api/analytics/heatmap${qs({ ...f })}`, {}, s),
  benchmark: (s?: AbortSignal) => request<{ results: BenchmarkRow[]; video?: string }>("/api/analytics/benchmark", {}, s),
  evaluation: (s?: AbortSignal) => request<Evaluation>("/api/analytics/evaluation", {}, s),
  liveStatus: (s?: AbortSignal) => request<RunStatus>("/api/live/status", {}, s),
  videos: (s?: AbortSignal) =>
    request<{ videos: string[]; live_sources: boolean; cameras: string[]; active_camera: string }>("/api/live/videos", {}, s),
  start: (body: { video: string; camera?: string; tracker?: string; model?: string; assume_signal?: string }) =>
    request<RunStatus>("/api/live/start", { method: "POST", body: JSON.stringify(body) }),
  stop: () => request<{ stopping: boolean }>("/api/live/stop", { method: "POST" }),
};

export interface SetupApproach {
  name: string;
  stop_line: [number, number][] | null;
  direction: "positive" | "negative";
  signal_roi: [number, number, number, number] | null;
}
export interface SetupCamera {
  id: string;
  exists: boolean;
  frame_size: [number, number] | null;
  speed: { method: string; lines?: [number, number][][]; distance_m?: number; image_points?: [number, number][]; world_points?: [number, number][] } | null;
  approaches: SetupApproach[];
  uses_legacy_single_line?: boolean;
}
export interface SetupSave {
  camera: string;
  approach?: string;
  frame_size: [number, number];
  stop_line?: number[][];
  direction?: "positive" | "negative";
  signal_roi?: number[];
  min_pixel_ratio?: number;
  speed?: { method: "lines" | "homography"; points: number[][]; distance_m?: number; rect_m?: number[] };
  clear_speed?: boolean;
}

export const setupApi = {
  camera: (id: string, s?: AbortSignal) => request<SetupCamera>(`/api/setup/camera/${encodeURIComponent(id)}`, {}, s),
  save: (body: SetupSave) => request<SetupCamera & { saved: string[] }>("/api/setup/save", { method: "POST", body: JSON.stringify(body) }),
  removeApproach: (camera: string, name: string) =>
    request<SetupCamera>(`/api/setup/approach${qs({ camera, name })}`, { method: "DELETE" }),
  /** One frame as an object URL plus the real frame size (the image itself may be downscaled). */
  async frame(source: string, index: number, signal?: AbortSignal): Promise<{ url: string; width: number; height: number }> {
    const token = getToken();
    const res = await fetch(`/api/setup/frame${qs({ source, index })}`, { signal, headers: token ? { Authorization: `Bearer ${token}` } : {} });
    if (res.status === 401) window.dispatchEvent(new Event("tvd-unauthorized"));
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
      throw new ApiError(res.status, String(detail));
    }
    const blob = await res.blob();
    return { url: URL.createObjectURL(blob), width: Number(res.headers.get("X-Frame-Width")), height: Number(res.headers.get("X-Frame-Height")) };
  },
};

export const exportUrl = {
  csv: (f: Filters) => withToken(`/api/events/export.csv${qs({ ...f })}`),
  pdf: (f: Filters) => withToken(`/api/report.pdf${qs({ ...f })}`),
};
export const streamUrl = () => withToken("/api/live/stream.mjpg");
export const evidenceUrl = (path: string) => withToken(path);
export function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return withToken(`${proto}://${location.host}/api/ws/live`);
}
