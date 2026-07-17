"use client";

import { create } from "zustand";
import { getJob, uploadPlan, type JobStatus } from "./api";

interface PlanState {
  jobId: string | null;
  job: JobStatus | null;
  uploading: boolean;
  error: string | null;
  upload: (file: File) => Promise<void>;
  reset: () => void;
}

const POLL_MS = 1500;

export const usePlanStore = create<PlanState>((set, get) => ({
  jobId: null,
  job: null,
  uploading: false,
  error: null,

  upload: async (file: File) => {
    set({ uploading: true, error: null, job: null, jobId: null });
    try {
      const jobId = await uploadPlan(file);
      set({ jobId, uploading: false });
      const poll = async () => {
        if (get().jobId !== jobId) return; // superseded by a newer upload
        try {
          const job = await getJob(jobId);
          set({ job });
          if (job.status === "queued" || job.status === "processing") {
            setTimeout(poll, POLL_MS);
          }
        } catch (err) {
          set({ error: err instanceof Error ? err.message : "Polling failed" });
        }
      };
      poll();
    } catch (err) {
      set({ uploading: false, error: err instanceof Error ? err.message : "Upload failed" });
    }
  },

  reset: () => set({ jobId: null, job: null, uploading: false, error: null }),
}));
