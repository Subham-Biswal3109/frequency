import { Database, Info, ShieldAlert, ShieldCheck, Shield } from "lucide-react";

import { Panel } from "@/components/wire/Panel";
import type { PredictRequest, PredictResponse } from "@/types/wire-watcher";
import { formatFrequencyRange, formatLocation, formatProbability } from "@/utils/format";
import { cn } from "@/lib/utils";

export function PredictionResultCard({
  result,
  request,
  receivedAt,
}: {
  result: PredictResponse;
  request: PredictRequest;
  receivedAt: string;
}) {
  const available = result.available;
  const ood = result.ood_warning;
  const confidence = result.confidence;

  const getConfidenceIcon = () => {
    if (ood || confidence === "Low" || confidence === "OOD / Unreliable") return <ShieldAlert className="size-4" />;
    if (confidence === "High") return <ShieldCheck className="size-4" />;
    return <Shield className="size-4" />;
  };

  const getConfidenceColor = () => {
    if (ood || confidence === "Low" || confidence === "OOD / Unreliable") return "text-destructive";
    if (confidence === "High") return "text-available";
    return "text-warning";
  };

  return (
    <div className="space-y-4">
      <section
        className={cn(
          "panel relative overflow-hidden px-5 py-6",
          available ? "border-available/35" : "border-occupied/35",
        )}
      >
        <span
          className={cn(
            "pointer-events-none absolute -right-16 -top-16 size-56 rounded-full blur-3xl",
            available ? "bg-available/12" : "bg-occupied/12",
          )}
        />
        <div className="flex justify-between items-start">
            <div>
                <p className="label-caps">Model output</p>
                <p
                  className={cn(
                    "font-display mt-2 text-4xl font-semibold tracking-tight sm:text-5xl",
                    available ? "text-available" : "text-occupied",
                  )}
                >
                  {available ? "AVAILABLE" : "NOT AVAILABLE"}
                </p>
            </div>
            <div className="text-right">
                <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface/50 px-2.5 py-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground shadow-sm">
                    Data Source: {result.data_source}
                </span>
            </div>
        </div>

        <dl className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <div>
            <dt className="label-caps">Probability</dt>
            <dd className="numeric mt-1 text-xl">{formatProbability(result.probability)}</dd>
          </div>
          <div>
            <dt className="label-caps">Confidence</dt>
            <dd className={cn("mt-1 text-lg font-semibold flex items-center gap-1.5", getConfidenceColor())}>
              {getConfidenceIcon()}
              {confidence}
            </dd>
          </div>
          <div>
            <dt className="label-caps">Threshold</dt>
            <dd className="numeric mt-1 text-xl">{result.threshold !== undefined ? `${(result.threshold * 100).toFixed(1)}%` : "N/A"}</dd>
          </div>
          <div>
            <dt className="label-caps">SNR</dt>
            <dd className="numeric mt-1 text-xl">{request.snr_db} dB</dd>
          </div>
          <div>
            <dt className="label-caps">Signal Power</dt>
            <dd className="numeric mt-1 text-xl">{request.signal_power_dbm} dBm</dd>
          </div>
          <div>
            <dt className="label-caps">Noise Floor</dt>
            <dd className="numeric mt-1 text-xl">{request.noise_floor_dbm} dBm</dd>
          </div>
        </dl>
      </section>

      {result.ood_warning && (
        <div className="rounded-lg border border-destructive/35 bg-destructive/8 px-4 py-3 text-sm">
          <p className="flex items-center gap-2 font-semibold text-destructive">
            <Info className="size-4" /> Out-of-Distribution Warning
          </p>
          <p className="mt-1 text-foreground/85">{result.warning}</p>
        </div>
      )}

      {result.important_features && result.important_features.length > 0 && (
        <Panel title="Prediction factors" subtitle="These are the most important model features driving this prediction, extracted from model metadata.">
          <div className="grid gap-x-6 gap-y-2 sm:grid-cols-3">
            {result.important_features.map((feat, i) => (
              <div key={feat} className="flex items-baseline gap-2 border-b border-border/50 py-1.5">
                <span className="text-xs text-muted-foreground">{i + 1}.</span>
                <span className="text-sm font-medium">{feat}</span>
              </div>
            ))}
          </div>
        </Panel>
      )}

      <p className="flex items-start gap-2 text-xs text-muted-foreground">
        <Info className="mt-0.5 size-3.5 shrink-0" />
        Output is an ML-based availability prediction trained on synthetic data — not a real-time
        spectrum measurement or a guarantee of availability.
      </p>
    </div>
  );
}
