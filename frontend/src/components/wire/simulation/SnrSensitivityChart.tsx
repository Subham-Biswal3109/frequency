import { Loader2, Play } from "lucide-react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Button } from "@/components/ui/button";
import { Panel } from "@/components/wire/Panel";
import type { SnrSweepPoint } from "@/types/wire-watcher";

export function SnrSensitivityChart({
  points,
  onRun,
  pending,
}: {
  points: SnrSweepPoint[] | null;
  onRun: () => void;
  pending: boolean;
}) {
  return (
    <Panel
      title="SNR Sensitivity Experiment"
      subtitle="Sweeps SNR from 0–30 dB and calls the existing model at each point — every value shown is a real model prediction, not interpolated."
      action={
        <Button variant="outline" size="sm" onClick={onRun} disabled={pending}>
          {pending ? <Loader2 className="mr-1.5 size-3.5 animate-spin" /> : <Play className="mr-1.5 size-3.5" />}
          Run Sweep
        </Button>
      }
    >
      {!points ? (
        <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">
          Run the sweep to see how the existing model's availability probability changes with SNR.
        </div>
      ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={points.map((p) => ({ snr: p.snr_db, probability: (p.probability ?? 0) * 100 }))}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" vertical={false} />
              <XAxis dataKey="snr" tickFormatter={(v) => `${v} dB`} fontSize={11} className="fill-muted-foreground" />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} fontSize={11} className="fill-muted-foreground" />
              <Tooltip
                formatter={(value: number) => [`${value.toFixed(1)}%`, "Availability probability"]}
                labelFormatter={(v) => `SNR: ${v} dB`}
              />
              <Line type="monotone" dataKey="probability" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {points?.some((p) => p.ood_warning) ? (
        <p className="mt-2 text-xs text-warning">
          ⚠ Some sweep points fall outside the model's training distribution (OOD) — see the underlying table for details.
        </p>
      ) : null}
    </Panel>
  );
}
