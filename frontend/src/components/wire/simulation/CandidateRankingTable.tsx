import { cn } from "@/lib/utils";
import type { AllocationCandidate } from "@/types/wire-watcher";

export function CandidateRankingTable({
  candidates,
  selectedRank,
}: {
  candidates: AllocationCandidate[];
  selectedRank?: number | undefined;
}) {
  if (candidates.length === 0) {
    return (
      <div className="panel px-4 py-6 text-center text-sm text-muted-foreground">
        No candidate channels satisfied the requested bandwidth.
      </div>
    );
  }

  return (
    <div className="panel px-4 py-4 lg:px-5">
      <h3 className="mb-3 text-sm font-semibold">Channel Ranking</h3>
      <div className="-mx-4 overflow-x-auto px-4">
        <table className="w-full min-w-[600px] border-separate border-spacing-0 text-sm">
          <thead>
            <tr className="label-caps text-left">
              <th className="border-b border-border pb-2 pr-4 font-normal">Rank</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Channels</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Frequency</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Bandwidth</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Avg SNR</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">ML Availability</th>
              <th className="border-b border-border pb-2 font-normal">Score</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((c) => (
              <tr
                key={c.channel_ids.join("-")}
                className={cn(
                  "transition-colors hover:bg-surface-raised/50",
                  c.rank === selectedRank && "bg-primary/8",
                )}
              >
                <td className="numeric border-b border-border/60 py-2.5 pr-4">
                  {c.rank === selectedRank ? (
                    <span className="rounded-full border border-primary/40 bg-primary/12 px-2 py-0.5 text-xs text-primary">
                      #{c.rank} SELECTED
                    </span>
                  ) : (
                    c.rank
                  )}
                </td>
                <td className="numeric border-b border-border/60 py-2.5 pr-4">{c.channel_ids.join(", ")}</td>
                <td className="numeric border-b border-border/60 py-2.5 pr-4 whitespace-nowrap">
                  {c.start_mhz}–{c.end_mhz} MHz
                </td>
                <td className="numeric border-b border-border/60 py-2.5 pr-4">{c.total_bandwidth_mhz} MHz</td>
                <td className="numeric border-b border-border/60 py-2.5 pr-4">{c.avg_snr_db.toFixed(1)} dB</td>
                <td className="numeric border-b border-border/60 py-2.5 pr-4">
                  {c.avg_ml_probability !== null ? `${(c.avg_ml_probability * 100).toFixed(1)}%` : "—"}
                </td>
                <td className="numeric border-b border-border/60 py-2.5">{c.score.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
