export type EventType = "red_light_violation" | "amber_crossing" | "legal_crossing" | "overspeed";
export type SignalState = "red" | "yellow" | "green" | "unknown";

export interface EventRow {
  event_id: string;
  track_id: number;
  vehicle_class: string;
  event_type: EventType;
  signal_state: SignalState;
  frame_no: number;
  video_timestamp: string;
  wall_clock_time: string;
  confidence: number;
  plate_text: string | null;
  plate_conf: number | null;
  evidence_path: string | null;
  camera_id?: string | null;
  speed_kmh?: number | null;
  limit_kmh?: number | null;
  approach?: string | null;
  time: string | null;
  has_evidence: boolean;
}

export interface Facets {
  vehicle_class: string[];
  signal_state: string[];
  event_type: string[];
  camera_id: string[];
}

export interface EventList {
  total: number;
  items: EventRow[];
  facets: Facets;
}

export interface EventDetail extends EventRow {
  evidence: Record<string, string>;
  metadata: { plate?: { text?: string; conf?: number; format?: string; note?: string }; frames?: Record<string, number | null> };
}

export interface Kpis {
  vehicles_tracked: number;
  crossings: number;
  red_light_violations: number;
  amber_crossings: number;
  legal_crossings: number;
  overspeed_events: number;
  max_speed_kmh: number;
  plates_read: number;
  fps: number;
  status: string;
}

export interface TimePoint {
  t: string;
  red_light_violation: number;
  amber_crossing: number;
  legal_crossing: number;
  overspeed: number;
}

export interface NamedValue { name: string; value: number }
export interface HeatCell { day: number; hour: number; count: number }

export interface BenchmarkRow {
  model: string;
  tracker: string;
  fps?: number | null;
  infer_ms?: number | null;
  frames?: number | null;
  detections?: number | null;
  unique_ids?: number | null;
  mean_track_len?: number | null;
  short_tracks?: number | null;
  crossings?: number | null;
  violations?: number | null;
  error?: string;
}

export interface Metric { tp: number; fp: number; fn: number; precision: number | null; recall: number | null; f1: number | null }
export interface Evaluation {
  overall: Metric | null;
  per_type?: Record<string, Metric | null>;
  ground_truth_events?: number;
  predicted_events?: number;
  tolerance_frames?: number;
  mean_abs_frame_error?: number | null;
}

export interface RunStatus {
  state: "idle" | "starting" | "running" | "finished" | "error";
  video: string | null;
  error: string;
  summary: Record<string, number | string> | null;
  stats: Partial<LiveStats>;
}

export interface LiveStats {
  status: string;
  frame_no: number;
  total_frames: number;
  video_timestamp: string;
  signal_state: SignalState;
  signals?: Record<string, SignalState>;   // one entry per stop line (approach) when a camera has several
  vehicles_tracked: number;
  active_tracks: number;
  crossings: number;
  red_light_violations: number;
  amber_crossings: number;
  legal_crossings: number;
  overspeed_events: number;
  max_speed_kmh: number;
  plates_read: number;
  fps: number;
  camera_fps?: number;                      // what the camera really delivers
  cpu_percent?: number | null;              // whole-machine CPU use
  timings_ms?: { wait_ms?: number; detect_ms?: number; busy_ms?: number };   // smoothed per-frame costs
}

export interface LiveTick {
  type: "tick";
  run: Omit<RunStatus, "stats">;
  stats: Partial<LiveStats>;
  kpis: Kpis;
  new_events: EventRow[];
}

export interface Filters {
  q?: string;
  event_type?: string;
  vehicle_class?: string;
  signal_state?: string;
  plate?: string;
  date_from?: string;
  date_to?: string;
}
