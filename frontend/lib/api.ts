const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface RoomInfo {
  id: number;
  label: string;
  area_m2: number;
}

export interface FurnitureInfo {
  room_id: number;
  room_label: string;
  item: string;
  footprint_m2: number;
}

export interface Reports {
  room_schedule: {
    id: number;
    label: string;
    area_m2: number;
    perimeter_m: number;
    connected_rooms: number[];
  }[];
  materials: Record<string, number>;
  cost_estimate: {
    currency: string;
    items: Record<string, number>;
    total: number;
    disclaimer: string;
  };
}

export interface JobResult {
  rooms: RoomInfo[];
  adjacency: { a: number; b: number; opening: boolean }[];
  validation: Record<string, unknown> & { passed: boolean };
  stats: Record<string, unknown>;
  furniture: FurnitureInfo[];
  reports: Reports;
}

export const EXPORT_FORMATS = ["glb", "obj", "stl", "ply"] as const;
export type ExportFormat = (typeof EXPORT_FORMATS)[number];

export interface JobStatus {
  job_id: string;
  status: "queued" | "processing" | "done" | "failed";
  error: string | null;
  result: JobResult | null;
}

export async function uploadPlan(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_BASE}/api/v1/plans`, { method: "POST", body: form });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => null);
    throw new Error(detail?.detail ?? `Upload failed (${resp.status})`);
  }
  const data = await resp.json();
  return data.job_id as string;
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const resp = await fetch(`${API_BASE}/api/v1/jobs/${encodeURIComponent(jobId)}`);
  if (!resp.ok) throw new Error(`Job lookup failed (${resp.status})`);
  return resp.json();
}

export function modelUrl(jobId: string, fmt: ExportFormat = "glb"): string {
  return `${API_BASE}/api/v1/jobs/${encodeURIComponent(jobId)}/model.${fmt}`;
}
