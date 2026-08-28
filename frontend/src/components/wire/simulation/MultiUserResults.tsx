import { CheckCircle2, XCircle } from "lucide-react";

import { cn } from "@/lib/utils";
import type { MultiUserAllocation } from "@/types/wire-watcher";

export function MultiUserResults({ allocation }: { allocation: MultiUserAllocation }) {
  return (
    <div className="panel px-4 py-4 lg:px-5">
      <h3 className="mb-3 text-sm font-semibold">Multi-User Allocation — Sequential Results</h3>
      <div className="space-y-3">
        {allocation.user_results.map((r, i) => (
          <div
            key={`${r.user_id}-${i}`}
            className={cn(
              "flex items-start gap-3 rounded-md border px-3 py-3",
              r.success ? "border-available/30 bg-available/6" : "border-occupied/30 bg-occupied/6",
            )}
          >
            {r.success ? (
              <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-available" />
            ) : (
              <XCircle className="mt-0.5 size-4 shrink-0 text-occupied" />
            )}
            <div className="flex-1">
              <p className="text-sm font-semibold">
                {r.user_id} <span className="text-muted-foreground">— requested {r.requested_bandwidth_mhz} MHz</span>
              </p>
              {r.success && r.selected ? (
                <p className="numeric mt-1 text-xs text-muted-foreground">
                  Allocated Ch {r.selected.channel_ids.join(", ")} ({r.selected.start_mhz}–{r.selected.end_mhz} MHz)
                </p>
              ) : (
                <p className="mt-1 text-xs text-muted-foreground">{r.message}</p>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4">
        <p className="label-caps mb-2">Remaining Available Spectrum (after each user)</p>
        <div className="flex flex-wrap gap-2">
          {allocation.utilization_timeline.map((u, i) => (
            <span
              key={i}
              className="numeric rounded-full border border-border bg-surface/60 px-2.5 py-1 text-xs"
            >
              {i === 0 ? "Initial" : `After ${allocation.user_results[i - 1]?.user_id ?? `user ${i}`}`}: {u.available_mhz} MHz free
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
