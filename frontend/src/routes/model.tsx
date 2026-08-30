import { createFileRoute, Link } from "@tanstack/react-router";
import { ShieldAlert } from "lucide-react";

import { AppShell } from "@/components/layout/AppShell";
import { ApiErrorNotice } from "@/components/wire/ApiStateNotice";
import { Panel } from "@/components/wire/Panel";
import { useHealth, useJammingModelInfo } from "@/hooks/use-wire-watcher";

export const Route = createFileRoute("/model")({
  head: () => ({
    meta: [
      { title: "ML Model — Wire Watcher" },
      {
        name: "description",
        content:
          "Details of the Wire Watcher Random Forest spectrum availability model: features used, training data and reported load status.",
      },
      { property: "og:title", content: "ML Model — Wire Watcher" },
      {
        property: "og:description",
        content:
          "Random Forest model architecture, input features and load status for the Wire Watcher spectrum prediction system.",
      },
    ],
  }),
  component: ModelPage,
});

const FEATURES = [
  ["start_frequency_mhz", "Lower bound of the requested band (MHz)"],
  ["end_frequency_mhz", "Upper bound of the requested band (MHz)"],
  ["bandwidth_mhz", "Channel bandwidth (MHz)"],
  ["hour_of_day", "Hour of day, 0–23"],
  ["day_of_week", "Day of week, 0 = Monday"],
  ["signal_power_dbm", "Measured signal power (dBm)"],
  ["noise_floor_dbm", "Noise floor (dBm)"],
  ["snr_db", "Signal-to-noise ratio (dB)"],
  ["state", "State / province"],
  ["city", "City"],
  ["service_type", "Service category of the band"],
] as const;

function ModelPage() {
  const { data, error, isPending } = useHealth();
  const jamming = useJammingModelInfo();

  return (
    <AppShell
      title="ML Model"
      description="Two separate models power Wire Watcher: a spectrum availability predictor (below) and a research-only RF jamming detector (further down). They are never combined."
    >
      <div className="space-y-6">
        <Panel title="Spectrum Availability Model" subtitle="Purpose: Available / Occupied prediction.">
          <p className="text-sm text-muted-foreground">
            The primary model used across Prediction, Prediction History, and Spectrum Simulation. See
            the architecture and input features below.
          </p>
        </Panel>

        <div className="grid gap-6 xl:grid-cols-2">
          <Panel title="Architecture" subtitle="As implemented in the existing backend.">
            <dl className="grid gap-3 text-sm">
              <div className="flex justify-between gap-4">
                <dt className="text-muted-foreground">Algorithm</dt>
                <dd>Random Forest classifier</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-muted-foreground">Task</dt>
                <dd>Binary spectrum availability</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-muted-foreground">Serving</dt>
                <dd className="numeric">POST /api/predict</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-muted-foreground">Storage</dt>
                <dd>MySQL prediction log</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-muted-foreground">Training data</dt>
                <dd>Synthetic dataset</dd>
              </div>
            </dl>
            <p className="mt-4 text-xs text-warning">
              Accuracy, precision, recall and feature importances are not shown because the backend
              does not currently expose them. Add GET /api/model-info to display real metrics.
            </p>
          </Panel>

          <Panel title="Model status" subtitle="Reported by GET /api/health.">
            {error ? (
              <ApiErrorNotice
                error={error}
                endpoint="GET /api/health"
                purpose="Reports whether the trained Random Forest model is loaded in the Flask process and whether MySQL is reachable."
              />
            ) : (
              <dl className="grid gap-3 text-sm">
                <div className="flex justify-between gap-4">
                  <dt className="text-muted-foreground">Model loaded</dt>
                  <dd className="numeric">
                    {isPending
                      ? "…"
                      : typeof data?.model_loaded === "boolean"
                        ? data.model_loaded
                          ? "Yes"
                          : "No"
                        : (data?.model ?? "—")}
                  </dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-muted-foreground">API status</dt>
                  <dd className="numeric">{isPending ? "…" : (data?.status ?? data?.api ?? "—")}</dd>
                </div>
              </dl>
            )}
          </Panel>
        </div>

        <Panel
          title="Input features"
          subtitle="Exact field names accepted by POST /api/predict — never renamed by the frontend."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="label-caps py-2 pr-4 font-medium">Field</th>
                  <th className="label-caps py-2 font-medium">Meaning</th>
                </tr>
              </thead>
              <tbody>
                {FEATURES.map(([field, meaning]) => (
                  <tr key={field} className="border-b border-border/50 last:border-0">
                    <td className="numeric py-2 pr-4">{field}</td>
                    <td className="py-2 text-muted-foreground">{meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <hr className="border-border" />

        <Panel
          title="RF Interference / Jamming Detection Model"
          subtitle="Purpose: Benign / Malicious RF classification — a SEPARATE model and task from spectrum availability."
          action={
            <Link to="/jamming" className="text-xs font-medium text-primary hover:underline">
              Open detector →
            </Link>
          }
        >
          {jamming.isError ? (
            <ApiErrorNotice
              error={jamming.error}
              endpoint="GET /api/jamming/model-info"
              purpose="Reports the jamming detector's dataset, controlled metrics, and limitations."
            />
          ) : jamming.data ? (
            <div className="space-y-4">
              <div className="flex items-start gap-2 rounded-md border border-warning/30 bg-warning/8 px-3 py-2 text-xs text-warning">
                <ShieldAlert className="mt-0.5 size-4 shrink-0" />
                This model never predicts "Available" or "Occupied" — its two classes are benign vs.
                malicious RF activity.
              </div>
              <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                <div>
                  <dt className="label-caps text-[10px]">Dataset</dt>
                  <dd className="mt-0.5">{jamming.data.dataset_name}</dd>
                </div>
                <div>
                  <dt className="label-caps text-[10px]">Dataset Type</dt>
                  <dd className="numeric mt-0.5">{jamming.data.dataset_type}</dd>
                </div>
                <div>
                  <dt className="label-caps text-[10px]">F1 (controlled)</dt>
                  <dd className="numeric mt-0.5 text-primary">{jamming.data.primary_controlled_metrics.f1.toFixed(3)}</dd>
                </div>
                <div>
                  <dt className="label-caps text-[10px]">ROC-AUC (controlled)</dt>
                  <dd className="numeric mt-0.5 text-primary">{jamming.data.primary_controlled_metrics.roc_auc.toFixed(3)}</dd>
                </div>
              </dl>
              <p className="text-xs text-muted-foreground">
                Controlled evaluation was designed to avoid the environment/label confounding present in
                the raw test split (see the detector page for the full breakdown and limitations).
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Loading…</p>
          )}
        </Panel>
      </div>
    </AppShell>
  );
}
