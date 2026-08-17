import { createFileRoute } from "@tanstack/react-router";
import { RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { AppShell } from "@/components/layout/AppShell";
import { ApiErrorNotice } from "@/components/wire/ApiStateNotice";
import { Panel } from "@/components/wire/Panel";
import { useHealth, usePredictions, useModelInfo } from "@/hooks/use-wire-watcher";
import { API_BASE_URL } from "@/services/api";

export const Route = createFileRoute("/status")({
  head: () => ({
    meta: [
      { title: "System Status — Wire Watcher" },
      {
        name: "description",
        content:
          "Connectivity status of the Wire Watcher Flask REST API, Random Forest model and MySQL database.",
      },
      { property: "og:title", content: "System Status — Wire Watcher" },
      {
        property: "og:description",
        content:
          "Live reachability checks for the Wire Watcher backend: Flask API, model load state and database connection.",
      },
    ],
  }),
  component: StatusPage,
});

function StatusDot({ tone }: { tone: "ok" | "bad" | "unknown" }) {
  const color =
    tone === "ok"
      ? "bg-available"
      : tone === "bad"
        ? "bg-occupied"
        : "bg-muted-foreground";
  return <span className={`inline-block size-2 rounded-full ${color}`} />;
}

function Row({
  label,
  value,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  tone: "ok" | "bad" | "unknown" | "neutral";
}) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border/50 py-3 last:border-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="flex items-center gap-2 text-sm">
        {tone !== "neutral" && <StatusDot tone={tone} />}
        <span className="numeric font-medium">{value}</span>
      </span>
    </div>
  );
}

function StatusPage() {
  const health = useHealth();
  const predictions = usePredictions();
  const modelInfo = useModelInfo();

  const apiTone = health.error ? "bad" : health.data ? "ok" : "unknown";
  const dbFlag = health.data?.database_connected;
  const dbTone =
    typeof dbFlag === "boolean" ? (dbFlag ? "ok" : "bad") : predictions.error ? "bad" : predictions.data ? "ok" : "unknown";
  const modelFlag = health.data?.model_loaded;
  const modelTone = typeof modelFlag === "boolean" ? (modelFlag ? "ok" : "bad") : "unknown";

  return (
    <AppShell
      title="System Status"
      description="Reachability of the existing Flask + Random Forest + MySQL stack, as reported by the backend itself."
    >
      <div className="space-y-6">
        {health.error ? (
          <ApiErrorNotice
            error={health.error}
            endpoint="GET /api/health"
            purpose="Single health endpoint returning API status, model load state and database connectivity."
          />
        ) : null}

        <Panel
          title="Components"
          subtitle="Unknown values stay blank — nothing is assumed to be healthy."
          action={
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                void health.refetch();
                void predictions.refetch();
                void modelInfo.refetch();
              }}
              disabled={health.isFetching || predictions.isFetching || modelInfo.isFetching}
            >
              <RefreshCw
                className={health.isFetching || predictions.isFetching || modelInfo.isFetching ? "size-3 animate-spin" : "size-3"}
              />
              Re-check
            </Button>
          }
        >
          <Row
            label="Flask REST API"
            value={
              health.isPending
                ? "checking…"
                : health.error
                  ? "unreachable"
                  : (health.data?.status ?? health.data?.api ?? "responding")
            }
            tone={apiTone}
          />
          <Row
            label="MySQL database"
            value={
              typeof dbFlag === "boolean"
                ? dbFlag
                  ? "connected"
                  : "disconnected"
                : predictions.isPending
                  ? "checking…"
                  : predictions.error
                    ? "records unavailable"
                    : "records readable"
            }
            tone={dbTone}
          />
          <Row
            label="ML Model Loader"
            value={
              health.isPending
                ? "checking…"
                : typeof modelFlag === "boolean"
                  ? modelFlag
                    ? "loaded"
                    : "not loaded"
                  : "not reported"
            }
            tone={modelTone}
          />
        </Panel>

        <Panel title="Model Metadata" subtitle="Model configuration reported directly by the backend.">
          <Row label="Model Version" value={modelInfo.data?.model_version ?? "Unknown"} tone="neutral" />
          <Row label="Algorithm" value={modelInfo.data?.algorithm ?? "Unknown"} tone="neutral" />
          <Row label="Training Dataset" value={modelInfo.data?.dataset_type ?? "Unknown"} tone="neutral" />
          <Row label="Training Samples" value={modelInfo.data?.training_samples?.toLocaleString() ?? "Unknown"} tone="neutral" />
          <Row label="Real RF Validation" value={modelInfo.data?.real_rf_validation === false ? "Not available" : (modelInfo.data?.real_rf_validation ? "Yes" : "Unknown")} tone="neutral" />
          <Row label="Model Threshold" value={modelInfo.data?.best_threshold?.toFixed(2) ?? "0.50"} tone="neutral" />
        </Panel>

        <Panel title="Configuration" subtitle="Frontend-side connection settings.">
          <Row label="API base URL (VITE_API_BASE_URL)" value={API_BASE_URL} tone="neutral" />
          <Row label="Health poll interval" value="30 s" tone="neutral" />
        </Panel>
      </div>
    </AppShell>
  );
}
