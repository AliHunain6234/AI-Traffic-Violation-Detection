import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { wsUrl } from "@/lib/api";
import type { EventRow, Kpis, LiveTick } from "@/lib/types";

interface LiveState {
  connected: boolean;
  tick: LiveTick | null;
  kpis: Kpis | null;
  recent: EventRow[];      // newest first, capped
  eventVersion: number;    // increments whenever new events arrive; use to refresh lists
}

const Ctx = createContext<LiveState>({ connected: false, tick: null, kpis: null, recent: [], eventVersion: 0 });
const MAX_RECENT = 25;

/** One shared WebSocket for the whole app, with exponential-backoff reconnects. */
export function LiveProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<LiveState>({ connected: false, tick: null, kpis: null, recent: [], eventVersion: 0 });
  const retry = useRef(0);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let closed = false;

    const connect = () => {
      ws = new WebSocket(wsUrl());
      ws.onopen = () => { retry.current = 0; setState((s) => ({ ...s, connected: true })); };
      ws.onmessage = (msg) => {
        try {
          const tick = JSON.parse(msg.data as string) as LiveTick;
          setState((s) => ({
            connected: true,
            tick,
            kpis: tick.kpis,
            recent: tick.new_events.length ? [...tick.new_events.slice().reverse(), ...s.recent].slice(0, MAX_RECENT) : s.recent,
            eventVersion: s.eventVersion + (tick.new_events.length ? 1 : 0),
          }));
        } catch { /* ignore malformed frames */ }
      };
      ws.onclose = () => {
        setState((s) => ({ ...s, connected: false }));
        if (!closed) {
          const delay = Math.min(15000, 1000 * 2 ** retry.current++);
          timer = setTimeout(connect, delay);
        }
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => { closed = true; clearTimeout(timer); ws?.close(); };
  }, []);

  const value = useMemo(() => state, [state]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useLive = () => useContext(Ctx);
