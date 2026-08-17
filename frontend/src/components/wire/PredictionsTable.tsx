import { AvailabilityBadge } from "@/components/wire/AvailabilityBadge";
import type { PredictionRecord } from "@/types/wire-watcher";
import {
  formatFrequencyRange,
  formatLocation,
  formatNumber,
  formatProbability,
  formatTimestamp,
} from "@/utils/format";

export function PredictionsTable({
  records,
  showBandwidth = true,
}: {
  records: PredictionRecord[];
  showBandwidth?: boolean;
}) {
  return (
    <>
      {/* Desktop / tablet table */}
      <div className="-mx-4 hidden overflow-x-auto px-4 md:block">
        <table className="w-full min-w-[720px] border-separate border-spacing-0 text-sm">
          <thead>
            <tr className="label-caps text-left">
              <th className="border-b border-border pb-2 pr-4 font-normal">Frequency</th>
              {showBandwidth ? (
                <th className="border-b border-border pb-2 pr-4 font-normal">Bandwidth</th>
              ) : null}
              <th className="border-b border-border pb-2 pr-4 font-normal">Location</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Prediction</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Probability</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">SNR</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Power</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Noise Floor</th>
              <th className="border-b border-border pb-2 pr-4 font-normal">Source</th>
              <th className="border-b border-border pb-2 font-normal">Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r) => (
              <tr key={r.id} className="transition-colors hover:bg-surface-raised/50">
                <td className="numeric border-b border-border/60 py-3 pr-4 whitespace-nowrap">
                  {formatFrequencyRange(r.start_frequency_mhz, r.end_frequency_mhz)}
                </td>
                {showBandwidth ? (
                  <td className="numeric border-b border-border/60 py-3 pr-4 whitespace-nowrap">
                    {formatNumber(r.bandwidth_mhz, 0, " MHz")}
                  </td>
                ) : null}
                <td className="border-b border-border/60 py-3 pr-4">
                  {formatLocation(r.city, r.state)}
                  {r.service_type ? (
                    <span className="block text-xs text-muted-foreground">{r.service_type}</span>
                  ) : null}
                </td>
                <td className="border-b border-border/60 py-3 pr-4">
                  <AvailabilityBadge available={r.available} />
                </td>
                <td className="numeric border-b border-border/60 py-3 pr-4">
                  {formatProbability(r.probability)}
                </td>
                <td className="numeric border-b border-border/60 py-3 pr-4">
                  {formatNumber(r.snr_db, 1, " dB")}
                </td>
                <td className="numeric border-b border-border/60 py-3 pr-4 whitespace-nowrap">
                  {formatNumber(r.signal_power_dbm, 1, " dBm")}
                </td>
                <td className="numeric border-b border-border/60 py-3 pr-4 whitespace-nowrap">
                  {formatNumber(r.noise_floor_dbm, 1, " dBm")}
                </td>
                <td className="border-b border-border/60 py-3 pr-4 text-xs text-muted-foreground uppercase">
                  {r.data_source || "SYNTHETIC"}
                </td>
                <td className="numeric border-b border-border/60 py-3 text-xs text-muted-foreground whitespace-nowrap">
                  {formatTimestamp(r.timestamp)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile cards */}
      <ul className="space-y-3 md:hidden">
        {records.map((r) => (
          <li key={r.id} className="rounded-lg border border-border bg-surface/60 p-3">
            <div className="flex items-start justify-between gap-3">
              <p className="numeric text-sm">
                {formatFrequencyRange(r.start_frequency_mhz, r.end_frequency_mhz)}
              </p>
              <AvailabilityBadge available={r.available} />
            </div>
            <dl className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div>
                <dt className="label-caps">Bandwidth</dt>
                <dd className="numeric">{formatNumber(r.bandwidth_mhz, 0, " MHz")}</dd>
              </div>
              <div>
                <dt className="label-caps">Probability</dt>
                <dd className="numeric">{formatProbability(r.probability)}</dd>
              </div>
              <div>
                <dt className="label-caps">Location</dt>
                <dd>{formatLocation(r.city, r.state)}</dd>
              </div>
              <div>
                <dt className="label-caps">SNR</dt>
                <dd className="numeric">{formatNumber(r.snr_db, 1, " dB")}</dd>
              </div>
              <div>
                <dt className="label-caps">Source</dt>
                <dd>{r.data_source || "SYNTHETIC"}</dd>
              </div>
              <div>
                <dt className="label-caps">Timestamp</dt>
                <dd className="numeric">{formatTimestamp(r.timestamp)}</dd>
              </div>
            </dl>
          </li>
        ))}
      </ul>
    </>
  );
}
