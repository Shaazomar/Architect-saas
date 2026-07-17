"use client";

import dynamic from "next/dynamic";
import { useCallback, useRef, useState } from "react";
import { EXPORT_FORMATS, modelUrl } from "@/lib/api";
import { usePlanStore } from "@/lib/store";

const ModelViewer = dynamic(() => import("@/components/ModelViewer"), { ssr: false });

const MAX_UPLOAD_MB = 20;

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    queued: "bg-amber-500/15 text-amber-400",
    processing: "bg-sky-500/15 text-sky-400 animate-pulse",
    done: "bg-emerald-500/15 text-emerald-400",
    failed: "bg-red-500/15 text-red-400",
  };
  return (
    <span className={`rounded-full px-3 py-1 text-xs font-medium ${styles[status] ?? ""}`}>
      {status}
    </span>
  );
}

export default function Home() {
  const { job, jobId, uploading, error, upload } = usePlanStore();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [showRoof, setShowRoof] = useState(false);

  const handleFile = useCallback(
    (file: File | undefined) => {
      setLocalError(null);
      if (!file) return;
      if (!["image/png", "image/jpeg"].includes(file.type)) {
        setLocalError("Please choose a PNG or JPEG floor plan image.");
        return;
      }
      if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
        setLocalError(`File is larger than ${MAX_UPLOAD_MB} MB.`);
        return;
      }
      void upload(file);
    },
    [upload],
  );

  const ready = job?.status === "done" && jobId;
  const displayError = localError ?? error ?? (job?.status === "failed" ? job.error : null);

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 p-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Architect SaaS</h1>
          <p className="text-sm text-neutral-400">2D floor plan → BIM-ready 3D reconstruction</p>
        </div>
        {job && <StatusBadge status={job.status} />}
      </header>

      <section
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFile(e.dataTransfer.files[0]);
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          dragOver ? "border-sky-400 bg-sky-500/5" : "border-neutral-700 hover:border-neutral-500"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <p className="text-neutral-300">
          {uploading ? "Uploading…" : "Drop a floor plan (PNG/JPEG) here, or click to browse"}
        </p>
        <p className="mt-1 text-xs text-neutral-500">Max {MAX_UPLOAD_MB} MB</p>
      </section>

      {displayError && (
        <div className="rounded-lg border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {displayError}
        </div>
      )}

      <div className="grid flex-1 gap-6 lg:grid-cols-[2fr_1fr]">
        <section className="relative min-h-[480px] overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
          {ready && (
            <label className="absolute right-3 top-3 z-10 flex cursor-pointer items-center gap-2 rounded-lg bg-neutral-800/80 px-3 py-1.5 text-xs text-neutral-300 backdrop-blur">
              <input
                type="checkbox"
                checked={showRoof}
                onChange={(e) => setShowRoof(e.target.checked)}
                className="accent-sky-500"
              />
              Roof
            </label>
          )}
          {ready ? (
            <ModelViewer url={modelUrl(jobId)} showRoof={showRoof} />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-neutral-500">
              {job?.status === "processing" || job?.status === "queued"
                ? "Reconstructing 3D model…"
                : "The 3D model appears here after processing"}
            </div>
          )}
        </section>

        <aside className="flex flex-col gap-4">
          <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
            <h2 className="mb-3 text-sm font-medium text-neutral-300">Detected rooms</h2>
            {job?.result?.rooms?.length ? (
              <ul className="space-y-2">
                {job.result.rooms.map((room) => (
                  <li
                    key={room.id}
                    className="flex items-center justify-between rounded-lg bg-neutral-800/60 px-3 py-2 text-sm"
                  >
                    <span className="capitalize">{room.label.replace("_", " ")}</span>
                    <span className="text-neutral-400">{room.area_m2} m²</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-neutral-500">No analysis yet.</p>
            )}
          </section>

          {job?.result?.validation && (
            <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
              <h2 className="mb-3 text-sm font-medium text-neutral-300">Validation</h2>
              <ul className="space-y-1 text-sm">
                {Object.entries(job.result.validation)
                  .filter(([, v]) => typeof v === "boolean")
                  .map(([key, value]) => (
                    <li key={key} className="flex items-center justify-between">
                      <span className="text-neutral-400">{key.replaceAll("_", " ")}</span>
                      <span className={value ? "text-emerald-400" : "text-red-400"}>
                        {value ? "pass" : "fail"}
                      </span>
                    </li>
                  ))}
              </ul>
            </section>
          )}

          {job?.result?.furniture && job.result.furniture.length > 0 && (
            <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
              <h2 className="mb-3 text-sm font-medium text-neutral-300">
                Furniture ({job.result.furniture.length} items)
              </h2>
              <ul className="max-h-40 space-y-1 overflow-y-auto text-sm">
                {job.result.furniture.map((f, i) => (
                  <li key={i} className="flex items-center justify-between">
                    <span className="capitalize text-neutral-300">{f.item.replaceAll("_", " ")}</span>
                    <span className="text-neutral-500 capitalize">
                      {f.room_label.replaceAll("_", " ")}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {job?.result?.reports?.cost_estimate && (
            <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
              <h2 className="mb-3 text-sm font-medium text-neutral-300">Cost estimate</h2>
              <ul className="space-y-1 text-sm">
                {Object.entries(job.result.reports.cost_estimate.items).map(([key, value]) => (
                  <li key={key} className="flex items-center justify-between">
                    <span className="text-neutral-400 capitalize">{key.replaceAll("_", " ")}</span>
                    <span className="text-neutral-300">
                      {value.toLocaleString()} {job.result!.reports.cost_estimate.currency}
                    </span>
                  </li>
                ))}
                <li className="mt-2 flex items-center justify-between border-t border-neutral-800 pt-2 font-medium">
                  <span>Total</span>
                  <span className="text-emerald-400">
                    {job.result.reports.cost_estimate.total.toLocaleString()}{" "}
                    {job.result.reports.cost_estimate.currency}
                  </span>
                </li>
              </ul>
              <p className="mt-2 text-xs text-neutral-600">
                {job.result.reports.cost_estimate.disclaimer}
              </p>
            </section>
          )}

          {ready && (
            <div className="grid grid-cols-4 gap-2">
              {EXPORT_FORMATS.map((fmt) => (
                <a
                  key={fmt}
                  href={modelUrl(jobId, fmt)}
                  download={`model.${fmt}`}
                  className="rounded-lg bg-sky-600 px-2 py-2 text-center text-xs font-medium uppercase text-white transition-colors hover:bg-sky-500"
                >
                  {fmt}
                </a>
              ))}
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}
