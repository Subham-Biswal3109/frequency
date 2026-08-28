import { CheckCircle2, XCircle } from "lucide-react";

import { cn } from "@/lib/utils";
import type { AllocationCandidate } from "@/types/wire-watcher";

export function AllocationResultCard({
  success,
  selected,
  message,
  requestedBandwidth,
}: {
  success: boolean;
  selected: AllocationCandidate | null;
  message: string;
  requestedBandwidth: number;
}) {
  if (!success || !selected) {
    return (
      <section className="panel relative overflow-hidden border-occupied/35 px-5 py-6">
        <div className="flex items-start gap-3">
          <XCircle className="mt-0.5 size-6 shrink-0 text-occupied" />
          <div>
            <p className="label-caps">Simulated Allocation</p>
            <p className="font-display mt-1 text-2xl font-semibold tracking-tight text-occupied">
              NO CHANNEL ALLOCATED
            </p>
            <p className="mt-2 text-sm text-muted-foreground">{message}</p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="panel relative overflow-hidden border-primary/35 px-5 py-6">
      <span className="pointer-events-none absolute -right-16 -top-16 size-56 rounded-full bg-primary/12 blur-3xl" />
      <div className="flex items-start gap-3">
        <CheckCircle2 className="mt-0.5 size-6 shrink-0 text-primary" />
        <div className="w-full">
          <p className="label-caps">Simulated Allocation</p>
          <p className="font-display mt-1 text-3xl font-semibold tracking-tight text-primary sm:text-4xl">
            CHANNEL ALLOCATED
          </p>

          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Channel(s)" value={selected.channel_ids.join(", ")} />
            <Stat label="Frequency" value={`${selected.start_mhz}–${selected.end_mhz} MHz`} />
            <Stat label="Bandwidth" value={`${selected.total_bandwidth_mhz} MHz`} />
            <Stat
              label="Availability"
              value={selected.avg_ml_probability !== null ? `${(selected.avg_ml_probability * 100).toFixed(1)}%` : "—"}
            />
            <Stat label="Avg SNR" value={`${selected.avg_snr_db.toFixed(1)} dB`} />
            <Stat label="Requested" value={`${requestedBandwidth} MHz`} />
            <Stat label="Isolation Score" value={selected.isolation_score.toFixed(2)} />
            <Stat label="Overall Score" value={selected.score.toFixed(3)} />
          </div>

          <p className="mt-4 rounded-md border border-border bg-surface/60 px-3 py-2 text-sm text-foreground/85">
            {message}
          </p>
          <p className="mt-2 text-xs text-muted-foreground">
            Simulated allocation. This does not guarantee interference-free operation and is not a
            real-world spectrum allocation.
          </p>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="label-caps text-[10px]">{label}</p>
      <p className="numeric mt-0.5 text-sm font-semibold text-foreground">{value}</p>
    </div>
  );
}
