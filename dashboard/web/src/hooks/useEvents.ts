import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ListParams } from "@/lib/api";
import type { EventList, Filters } from "@/lib/types";

/** Generic fetch hook: aborts stale requests, exposes loading/error and a reload function. */
export function useFetch<T>(fetcher: (signal: AbortSignal) => Promise<T>, deps: unknown[], refreshKey = 0) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [tick, setTick] = useState(0);
  const fetchRef = useRef(fetcher);
  fetchRef.current = fetcher;

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    fetchRef.current(ctrl.signal)
      .then((d) => { setData(d); setError(""); })
      .catch((e: unknown) => {
        if ((e as Error).name !== "AbortError") setError(e instanceof Error ? e.message : "Request failed");
      })
      .finally(() => { if (!ctrl.signal.aborted) setLoading(false); });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, refreshKey, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, loading, error, reload };
}

/** Events list with debounced filters. `refreshKey` re-fetches when the live socket sees new events. */
export function useEvents(filters: Filters, params: ListParams, refreshKey = 0) {
  const [debounced, setDebounced] = useState(filters);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(filters), 250);
    return () => clearTimeout(id);
  }, [filters]);
  return useFetch<EventList>(
    (signal) => api.events(debounced, params, signal),
    [JSON.stringify(debounced), JSON.stringify(params)],
    refreshKey,
  );
}
