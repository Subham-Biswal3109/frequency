import { cn } from "@/lib/utils";
import type { SimulationChannel } from "@/types/wire-watcher";

const STATE_STYLES: Record<string, string> = {
  OCCUPIED: "bg-occupied/80",
  AVAILABLE: "bg-available/70",
  UNAVAILABLE: "bg-warning/70",
  ALLOCATED: "bg-primary/80",
};

export function ChannelOccupancyBar({ channels }: { channels: SimulationChannel[] }) {
  return (
    <div className="panel px-4 py-4 lg:px-5">
      <h3 className="mb-3 text-sm font-semibold">Channel Occupancy Map</h3>
      <div className="flex overflow-hidden rounded-md border border-border">
        {channels.map((c) => (
          <div
            key={c.channel_id}
            title={`Ch ${c.channel_id}: ${c.start_mhz}–${c.end_mhz} MHz — ${c.state}`}
            className={cn(
              "group relative flex h-10 flex-1 items-center justify-center border-r border-background/40 text-[10px] font-medium text-white/90 last:border-r-0",
              STATE_STYLES[c.state] ?? "bg-muted",
            )}
          >
            <span className="numeric">{c.channel_id}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap gap-4 text-xs text-muted-foreground">
        {["OCCUPIED", "AVAILABLE", "UNAVAILABLE", "ALLOCATED"].map((state) => (
          <span key={state} className="flex items-center gap-1.5">
            <span className={cn("size-2.5 rounded-sm", STATE_STYLES[state])} />
            {state}
          </span>
        ))}
      </div>
    </div>
  );
}
