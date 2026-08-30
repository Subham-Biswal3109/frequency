import { CheckCircle2, ShieldAlert, ShieldCheck, XCircle } from "lucide-react";

import { cn } from "@/lib/utils";
import type { JammingPredictResponse } from "@/types/wire-watcher";

export function JammingResultCard({ result }: { result: JammingPredictResponse }) {
  const isMalicious = result.prediction === "malicious";

  return (
    <section
      className={cn(
        "panel relative overflow-hidden px-5 py-6",
        isMalicious ? "border-occupied/35" : "border-available/35",
      )}
    >
      <div className="flex items-start gap-3">
        {isMalicious ? (
          <ShieldAlert className="mt-0.5 size-6 shrink-0 text-occupied" />
        ) : (
          <ShieldCheck className="mt-0.5 size-6 shrink-0 text-available" />
        )}
        <div className="w-full">
          <p className="label-caps">RF Interference / Jamming Detector</p>
          <p
            className={cn(
              "font-display mt-1 text-2xl font-semibold tracking-tight sm:text-3xl",
              isMalicious ? "text-occupied" : "text-available",
            )}
          >
            {isMalicious ? "Potential Jamming / Malicious RF Activity" : "Benign RF Activity"}
          </p>

          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
            <div>
              <p className="label-caps text-[10px]">Probability Malicious</p>
              <p className="numeric mt-0.5 text-sm font-semibold">{(result.probability_malicious * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="label-caps text-[10px]">Decision Threshold</p>
              <p className="numeric mt-0.5 text-sm font-semibold">{(result.threshold * 100).toFixed(0)}%</p>
            </div>
            {result.true_label ? (
              <div>
                <p className="label-caps text-[10px]">Ground Truth (held-out)</p>
                <p className="numeric mt-0.5 flex items-center gap-1 text-sm font-semibold">
                  {result.true_label}
                  {result.correct ? (
                    <CheckCircle2 className="size-3.5 text-available" />
                  ) : (
                    <XCircle className="size-3.5 text-occupied" />
                  )}
                </p>
              </div>
            ) : null}
          </div>

          <p className="mt-4 text-xs text-muted-foreground">{result.disclaimer}</p>
        </div>
      </div>
    </section>
  );
}
