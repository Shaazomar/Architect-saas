const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface RoomInfo {
  id: number;
  label: string;
  area_m2: number;
}

export interface JobResult {
  rooms: RoomInfo[];
  adjacency: { a: number; b: number; opening: boolean }[];
  validation: Record<string, unknown> & { passed: boolean };
  stats: Record<string, unknown>;
}

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

export function modelUrl(jobId: string): string {
  return `${API_BASE}/api/v1/jobs/${encodeURIComponent(jobId)}/model.glb`;
}
