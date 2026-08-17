import { AlertTriangle, PlugZap, Loader2 } from "lucide-react";

import { ApiError, describeError } from "@/services/api";
import { cn } from "@/lib/utils";

/** Explicit, honest empty/error states — the UI never silently falls back to fake data. */
export function ApiErrorNotice({
  error,
  endpoint,
  purpose,
  className,
}: {
  error: unknown;
  endpoint: string;
  purpose: string;
  className?: string;
}) {
  const info = describeError(error);
  const missing = error instanceof ApiError && error.kind === "not_implemented";

  return (
    <div
      className={cn(
        "rounded-lg border px-4 py-4 text-sm",
        missing
          ? "border-warning/40 bg-warning/8 text-warning-foreground/90"
          : "border-destructive/40 bg-destructive/8",
        className,
      )}
    >
      <p className="flex items-center gap-2 font-semibold">
        {missing ? <PlugZap className="size-4 text-warning" /> : <AlertTriangle className="size-4 text-destructive" />}
        <span className={missing ? "text-warning" : "text-destructive"}>{info.title}</span>
      </p>
      <p className="mt-2 text-foreground/85">{info.message}</p>
      {info.details?.length ? (
        <ul className="numeric mt-2 list-inside list-disc space-y-1 text-xs text-muted-foreground">
          {info.details.slice(0, 6).map((d, i) => (
            <li key={i}>{d}</li>
          ))}
        </ul>
      ) : null}
      <div className="mt-3 rounded-md border border-border bg-surface/70 px-3 py-2">
        <p className="label-caps">Required backend endpoint</p>
        <p className="numeric mt-1 text-xs text-foreground">{endpoint}</p>
        <p className="mt-1 text-xs text-muted-foreground">{purpose}</p>
      </div>
    </div>
  );
}

export function LoadingRow({ label = "Querying Flask API…" }: { label?: string }) {
  return (
    <p className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" /> {label}
    </p>
  );
}

export function EmptyRow({ label }: { label: string }) {
  return <p className="py-6 text-sm text-muted-foreground">{label}</p>;
}
