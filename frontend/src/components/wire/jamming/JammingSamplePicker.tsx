import { ShieldAlert, ShieldCheck } from "lucide-react";

import { cn } from "@/lib/utils";
import type { JammingSampleSummary } from "@/types/wire-watcher";

export function JammingSamplePicker({
  samples,
  selectedId,
  onSelect,
}: {
  samples: JammingSampleSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="max-h-96 space-y-1.5 overflow-y-auto pr-1">
      {samples.map((s) => (
        <button
          key={s.sample_id}
          type="button"
          onClick={() => onSelect(s.sample_id)}
          className={cn(
            "flex w-full items-center justify-between rounded-md border px-3 py-2 text-left text-sm transition-colors",
            s.sample_id === selectedId
              ? "border-primary/50 bg-primary/8"
              : "border-border hover:bg-surface-raised/50",
          )}
        >
          <div className="flex items-center gap-2 truncate">
            {s.true_label === "malicious" ? (
              <ShieldAlert className="size-4 shrink-0 text-occupied" />
            ) : (
              <ShieldCheck className="size-4 shrink-0 text-available" />
            )}
            <span className="numeric truncate text-xs">{s.file_name}</span>
          </div>
          <span className="ml-2 shrink-0 text-xs text-muted-foreground">
            {s.band} · {s.scan_mode}
            {s.waveform ? ` · ${s.waveform}` : ""}
          </span>
        </button>
      ))}
    </div>
  );
}
