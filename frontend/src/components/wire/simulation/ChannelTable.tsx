import { Download } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { SimulationChannel } from "@/types/wire-watcher";

const STATE_BADGE: Record<string, string> = {
  OCCUPIED: "border-occupied/40 bg-occupied/12 text-occupied",
  AVAILABLE: "border-available/40 bg-available/12 text-available",
  UNAVAILABLE: "border-warning/40 bg-warning/12 text-warning",
  ALLOCATED: "border-primary/40 bg-primary/12 text-primary",
};

function toCsv(channels: SimulationChannel[], simulationId: string): string {
  const headers = [
    "timestamp", "simulation_id", "channel_id", "start_mhz", "end_mhz", "bandwidth_mhz",
    "rf_signal_power_dbm", "rf_noise_floor_dbm", "rf_snr_db", "rf_state",
    "ml_probability", "ml_decision", "ml_ood_warning", "state",
  ];
  const now = new Date().toISOString();
  const rows = channels.map((c) =>
    [
      now, simulationId, c.channel_id, c.start_mhz, c.end_mhz, c.bandwidth_mhz,
      c.rf_signal_power_dbm, c.rf_noise_floor_dbm, c.rf_snr_db, c.rf_state,
      c.ml_probability ?? "", c.ml_decision, c.ml_ood_warning, c.state,
    ].join(","),
  );
  return [headers.join(","), ...rows].join("\n");
}

function downloadCsv(content: string, filename: string) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function ChannelTable({ channels }: { channels: SimulationChannel[] }) {
  const simulationId = `sim-${Date.now()}`;

  return (
    <div className="panel px-4 py-4 lg:px-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold">Frequency Sweep — Per-Channel Detail</h3>
        <Button
          variant="outline"
          size="sm"
          onClick={() => downloadCsv(toCsv(channels, simulationId), `${simulationId}.csv`)}
        >
          <Download className="mr-1.5 size-3.5" /> Export CSV
        </Button>
      </div>
      <div className="-mx-4 overflow-x-auto px-4">
        <table className="w-full min-w-[820px] border-separate border-spacing-0 text-sm">
          <thead>
            <tr className="label-caps text-left">
              <th className="border-b border-border pb-2 pr-4 font-normal">Ch</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Frequency</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Power</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Noise Floor</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">SNR</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">RF Sensing</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">ML Availability</th>
              <th className="border-b border-border pb-2 font-normal">Final State</th>
            </tr>
          </thead>
          <tbody>
            {channels.map((c) => (
              <tr key={c.channel_id} className="transition-colors hover:bg-surface-raised/50">
                <td className="numeric border-b border-border/60 py-2.5 pr-4">{c.channel_id}</td>
                <td className="numeric border-b border-border/60 py-2.5 pr-4 whitespace-nowrap">
                  {c.start_mhz}–{c.end_mhz} MHz
                </td>
                <td className="numeric border-b border-border/60 py-2.5 pr-4">{c.rf_signal_power_dbm.toFixed(1)} dBm</td>
                <td className="numeric border-b border-border/60 py-2.5 pr-4">{c.rf_noise_floor_dbm.toFixed(1)} dBm</td>
                <td className="numeric border-b border-border/60 py-2.5 pr-4">{c.rf_snr_db.toFixed(1)} dB</td>
                <td className="border-b border-border/60 py-2.5 pr-4">
                  <span className={cn("rounded-full border px-2 py-0.5 text-xs", STATE_BADGE[c.rf_state])}>
                    {c.rf_state}
                  </span>
                </td>
                <td className="numeric border-b border-border/60 py-2.5 pr-4">
                  {c.ml_probability !== null ? `${(c.ml_probability * 100).toFixed(1)}%` : "—"}
                  {c.ml_ood_warning ? <span className="ml-1 text-warning">⚠</span> : null}
                </td>
                <td className="border-b border-border/60 py-2.5">
                  <span className={cn("rounded-full border px-2 py-0.5 text-xs", STATE_BADGE[c.state])}>
                    {c.state}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
