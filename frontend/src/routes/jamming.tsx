import { createFileRoute } from "@tanstack/react-router";
import { ShieldAlert } from "lucide-react";
import { useState } from "react";

import { AppShell } from "@/components/layout/AppShell";
import { ApiErrorNotice } from "@/components/wire/ApiStateNotice";
import { Panel } from "@/components/wire/Panel";
import { JammingLimitationsNotice } from "@/components/wire/jamming/JammingLimitationsNotice";
import { JammingModelMetricsPanel } from "@/components/wire/jamming/JammingModelMetricsPanel";
import { JammingResultCard } from "@/components/wire/jamming/JammingResultCard";
import { JammingSamplePicker } from "@/components/wire/jamming/JammingSamplePicker";
import { useJammingModelInfo, useJammingSamples, usePredictJamming } from "@/hooks/use-wire-watcher";

export const Route = createFileRoute("/jamming")({
  head: () => ({
    meta: [
      { title: "RF Interference / Jamming Detection — Wire Watcher" },
      {
        name: "description",
        content:
          "A separate research model trained on real RF spectral-scan measurements to detect benign vs malicious (jamming) RF activity. Distinct from Wire Watcher's spectrum availability predictor.",
      },
      { property: "og:title", content: "RF Interference / Jamming Detection — Wire Watcher" },
      {
        property: "og:description",
        content:
          "Random Forest jamming detector evaluated with an environment-controlled methodology to avoid confounded results.",
      },
    ],
  }),
  component: JammingPage,
});

function JammingPage() {
  const modelInfo = useJammingModelInfo();
  const samples = useJammingSamples();
  const predict = usePredictJamming();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const handleSelect = (id: string) => {
    setSelectedId(id);
    predict.mutate(id);
  };

  return (
    <AppShell
      title="RF Interference / Jamming Detection"
      description="A separate research model — not a spectrum availability predictor. Trained on real experimental RF spectral-scan measurements to distinguish benign RF activity from active jamming."
    >
      <div className="space-y-6">
        <div className="flex items-start gap-2 rounded-md border border-warning/30 bg-warning/8 px-3 py-2 text-xs text-warning">
          <ShieldAlert className="mt-0.5 size-4 shrink-0" />
          <p>
            This is a <strong>separate model and task</strong> from Wire Watcher's spectrum availability
            predictor. It classifies <strong>benign vs. malicious (jamming) RF activity</strong> — it does
            not predict "Available" or "Occupied" spectrum, and its outputs should never be read that way.
          </p>
        </div>

        {modelInfo.isError ? (
          <ApiErrorNotice
            error={modelInfo.error}
            endpoint="GET /api/jamming/model-info"
            purpose="Reports the jamming detector's dataset provenance, controlled validation metrics, and limitations."
          />
        ) : null}

        {modelInfo.data ? (
          <>
            <Panel
              title="Model"
              subtitle={`${modelInfo.data.algorithm} — ${modelInfo.data.task}`}
            >
              <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                <div>
                  <dt className="label-caps text-[10px]">Dataset</dt>
                  <dd className="mt-0.5">{modelInfo.data.dataset_name}</dd>
                </div>
                <div>
                  <dt className="label-caps text-[10px]">Dataset Type</dt>
                  <dd className="numeric mt-0.5">{modelInfo.data.dataset_type}</dd>
                </div>
                <div>
                  <dt className="label-caps text-[10px]">Version</dt>
                  <dd className="numeric mt-0.5">{modelInfo.data.model_version}</dd>
                </div>
                <div>
                  <dt className="label-caps text-[10px]">Decision Threshold</dt>
                  <dd className="numeric mt-0.5">{(modelInfo.data.best_threshold * 100).toFixed(0)}%</dd>
                </div>
              </dl>
            </Panel>

            <JammingModelMetricsPanel info={modelInfo.data} />
            <JammingLimitationsNotice limitations={modelInfo.data.limitations} />
          </>
        ) : null}

        <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
          <Panel
            title="Held-Out Test Samples"
            subtitle="Pick a real, labeled sample the model never saw during training to see its prediction."
          >
            {samples.isError ? (
              <ApiErrorNotice
                error={samples.error}
                endpoint="GET /api/jamming/samples"
                purpose="Lists held-out RF captures with true labels for demonstrating the detector."
              />
            ) : samples.data ? (
              <JammingSamplePicker
                samples={samples.data.samples}
                selectedId={selectedId}
                onSelect={handleSelect}
              />
            ) : (
              <p className="text-sm text-muted-foreground">Loading samples…</p>
            )}
          </Panel>

          <div className="space-y-6">
            {predict.isError ? (
              <ApiErrorNotice
                error={predict.error}
                endpoint="POST /api/jamming/predict"
                purpose="Runs the jamming detector on a selected held-out RF capture."
              />
            ) : predict.data ? (
              <JammingResultCard result={predict.data} />
            ) : (
              <div className="panel flex h-48 items-center justify-center px-4 text-center text-sm text-muted-foreground">
                Select a sample on the left to run the detector on a real, held-out RF capture.
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
