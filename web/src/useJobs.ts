import { useEffect, useRef, useState } from "react";

import { cancelJob, getJob, getJobResult, listJobs, simulate, submitScan,
  type SimulationResponse } from "./api/client";
import type { JobSummary, ScanRequest, SimConfig } from "./api/types";

export function useJobs(onResult: (response: SimulationResponse) => void,
                        onError: (error: unknown) => void) {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = useRef<string | null>(null);
  const loaded = useRef<string | null>(null);
  const versions = useRef(new Map<string, number>());
  const callbacks = useRef({ onResult, onError });
  callbacks.current = { onResult, onError };

  function merge(item: JobSummary) {
    if ((versions.current.get(item.id) ?? -1) > item.version) return;
    versions.current.set(item.id, item.version);
    setJobs((current) => [item, ...current.filter((job) => job.id !== item.id)]);
    if (item.id === selected.current && item.kind === "single" && item.status === "completed"
        && loaded.current !== item.id) void loadResult(item.id);
  }

  function snapshot(all: JobSummary[]) {
    for (const item of all) {
      if ((versions.current.get(item.id) ?? -1) <= item.version) {
        versions.current.set(item.id, item.version);
      }
    }
    setJobs((current) => {
      const byId = new Map(current.map((item) => [item.id, item]));
      for (const item of all) {
        if (!byId.has(item.id) || item.version >= byId.get(item.id)!.version) byId.set(item.id, item);
      }
      return [...all.map((item) => byId.get(item.id)!),
        ...current.filter((item) => !all.some((fresh) => fresh.id === item.id))];
    });
  }

  async function loadResult(id: string) {
    if (loaded.current === id) return;
    loaded.current = id;
    try {
      const response = await getJobResult(id);
      if (selected.current === id) callbacks.current.onResult(response);
    } catch (error) {
      if (selected.current === id) {
        loaded.current = null;
        callbacks.current.onError(error);
      }
    }
  }

  function select(id: string) {
    selected.current = id;
    setSelectedId(id);
    loaded.current = null;
    const item = jobs.find((job) => job.id === id);
    if (item?.kind === "single" && item.status === "completed") void loadResult(id);
  }

  useEffect(() => {
    let active = true;
    let fallback: number | undefined;
    const refresh = () => void listJobs().then((all) => {
      if (!active) return;
      snapshot(all);
      const latest = all.find((item) => item.kind === "single");
      if (!selected.current && latest) {
        selected.current = latest.id;
        setSelectedId(latest.id);
        if (latest.status === "completed") void loadResult(latest.id);
      } else {
        const current = all.find((item) => item.id === selected.current);
        if (current?.status === "completed" && current.kind === "single"
            && loaded.current !== current.id) void loadResult(current.id);
      }
    }).catch((error: unknown) => { if (active) callbacks.current.onError(error); });
    const startFallback = () => {
      if (fallback !== undefined) return;
      fallback = window.setInterval(refresh, 5000);
    };
    void refresh();
    if (typeof EventSource === "undefined") {
      startFallback();
      return () => { active = false; window.clearInterval(fallback); };
    }
    const source = new EventSource("/api/jobs/events");
    source.addEventListener("snapshot", (event) => {
      if (!active) return;
      const all = JSON.parse((event as MessageEvent).data) as JobSummary[];
      snapshot(all);
      const current = all.find((item) => item.id === selected.current);
      if (current?.kind === "single" && current.status === "completed"
          && loaded.current !== current.id) void loadResult(current.id);
    });
    source.addEventListener("update", (event) => {
      if (active) merge(JSON.parse((event as MessageEvent).data) as JobSummary);
    });
    source.onopen = () => { window.clearInterval(fallback); fallback = undefined; };
    source.onerror = startFallback;
    return () => { active = false; source.close(); window.clearInterval(fallback); };
  }, []);

  async function submitSingle(config: SimConfig, force: boolean) {
    setSubmitting(true);
    try {
      const receipt = force ? await simulate(config, true) : await simulate(config);
      selected.current = receipt.id;
      setSelectedId(receipt.id);
      loaded.current = null;
      merge(await getJob(receipt.id));
    } finally { setSubmitting(false); }
  }

  async function submitSweep(request: ScanRequest) {
    const receipt = await submitScan(request);
    merge(await getJob(receipt.id));
  }

  async function stop(id: string) { merge(await cancelJob(id)); }

  return { jobs, submitting, selectedId, select, submitSingle, submitSweep, stop };
}
