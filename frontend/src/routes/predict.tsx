import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

import { AppShell } from "@/components/layout/AppShell";
import { ApiErrorNotice } from "@/components/wire/ApiStateNotice";
import { Panel } from "@/components/wire/Panel";
import { PredictionForm } from "@/components/wire/PredictionForm";
import { PredictionResultCard } from "@/components/wire/PredictionResultCard";
import { SpectrumChart } from "@/components/wire/SpectrumChart";
import { usePredict, useAnalyzeSpectrum } from "@/hooks/use-wire-watcher";
import type { PredictRequest } from "@/types/wire-watcher";

export const Route = createFileRoute("/predict")({
  head: () => ({
    meta: [
      { title: "Spectrum Prediction — Wire Watcher" },
      {
        name: "description",
        content:
          "Submit frequency, signal, time and location features to the Wire Watcher Random Forest model and read the availability prediction.",
      },
      { property: "og:title", content: "Spectrum Prediction — Wire Watcher" },
      {
        property: "og:description",
        content:
          "Run the Wire Watcher Random Forest spectrum availability model against your own band, signal and location inputs.",
      },
    ],
  }),
  component: PredictPage,
});

function PredictPage() {
  const mutation = usePredict();
  const analyzeMutation = useAnalyzeSpectrum();
  const [submitted, setSubmitted] = useState<PredictRequest | null>(null);
  const [receivedAt, setReceivedAt] = useState<string>("");
  const [analysisResult, setAnalysisResult] = useState<import("@/types/wire-watcher").AnalyzeSpectrumResponse | null>(null);

  return (
    <AppShell
      title="Spectrum Prediction"
      description="Feature inputs are sent unchanged to POST /api/predict — the model is the source of truth."
    >
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-6">
          <PredictionForm
            pending={mutation.isPending}
            analyzing={analyzeMutation.isPending}
            onSubmit={(input) => {
              setSubmitted(input);
              mutation.mutate(input, {
                onSuccess: () => setReceivedAt(new Date().toLocaleString()),
              });
            }}
            onAnalyze={(input) => {
              analyzeMutation.mutate(input, {
                onSuccess: (data) => setAnalysisResult(data),
              });
            }}
          />
          {analyzeMutation.error && (
            <ApiErrorNotice error={analyzeMutation.error} endpoint="POST /api/spectrum/analyze" purpose="Simulates an RF environment." />
          )}
          {analysisResult && (
            <SpectrumChart
              frequencies={analysisResult.spectrum_data.frequencies}
              powers={analysisResult.spectrum_data.power_dbm}
              noiseFloor={analysisResult.noise_floor_dbm}
              peaks={analysisResult.detected_signals}
              centerFreq={(analysisResult.frequency_range.start_mhz + analysisResult.frequency_range.end_mhz) / 2}
              bandwidth={analysisResult.frequency_range.end_mhz - analysisResult.frequency_range.start_mhz}
            />
          )}
        </div>

        <div className="space-y-4">
          {mutation.error ? (
            <ApiErrorNotice
              error={mutation.error}
              endpoint="POST /api/predict"
              purpose="Runs the leakage-safe Random Forest model and returns prediction, available, probability and features_used."
            />
          ) : null}

          {mutation.data && submitted ? (
            <PredictionResultCard
              result={mutation.data}
              request={submitted}
              receivedAt={receivedAt}
            />
          ) : !mutation.error ? (
            <Panel title="Result" subtitle="Awaiting a prediction request.">
              <div className="flex h-56 flex-col items-center justify-center gap-2 text-center">
                <p className="label-caps">No output yet</p>
                <p className="max-w-sm text-sm text-muted-foreground">
                  Fill in the band, signal, time and location features, then run the model. The
                  result card shows the exact terminology returned by the backend.
                </p>
              </div>
            </Panel>
          ) : null}
        </div>
      </div>
    </AppShell>
  );
}
