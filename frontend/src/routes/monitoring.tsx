import { createFileRoute } from "@tanstack/react-router";
import { RefreshCw } from "lucide-react";
import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Button } from "@/components/ui/button";
import { AppShell } from "@/components/layout/AppShell";
import { ApiErrorNotice, EmptyRow, LoadingRow } from "@/components/wire/ApiStateNotice";
import { KpiCard, Panel } from "@/components/wire/Panel";
import { PredictionsTable } from "@/components/wire/PredictionsTable";
import { usePredictions, useModelInfo } from "@/hooks/use-wire-watcher";
import { formatProbability } from "@/utils/format";

export const Route = createFileRoute("/monitoring")({
  head: () => ({
    meta: [
      { title: "Spectrum Monitoring — Wire Watcher" },
      {
        name: "description",
        content:
          "Track how Wire Watcher spectrum availability predictions evolve over time using records stored in the MySQL database.",
      },
      { property: "og:title", content: "Spectrum Monitoring — Wire Watcher" },
      {
        property: "og:description",
        content:
          "Availability trend and latest prediction activity derived from stored Wire Watcher records.",
      },
    ],
  }),
  component: MonitoringPage,
});

function MonitoringPage() {
  const { data, isPending, error, refetch, isFetching } = usePredictions();
  const modelInfo = useModelInfo();
  const records = data ?? [];
  const kpis = modelInfo.data?.kpis;

  const trend = useMemo(() => {
    const buckets = new Map<string, { hour: string; available: number; occupied: number }>();
    for (const record of records) {
      if (!record.timestamp) continue;
      const time = new Date(record.timestamp);
      if (Number.isNaN(time.getTime())) continue;
      const key = `${time.toISOString().slice(0, 13)}:00`;
      const bucket = buckets.get(key) ?? { hour: key.slice(5).replace("T", " "), available: 0, occupied: 0 };
      if (record.available === true) bucket.available += 1;
      else if (record.available === false) bucket.occupied += 1;
      buckets.set(key, bucket);
    }
    return [...buckets.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([, v]) => v);
  }, [records]);

  return (
    <AppShell
      title="Spectrum Monitoring"
      description="Derived entirely from stored records returned by GET /api/predictions — no simulated telemetry."
    >
      <div className="space-y-6">
        {error ? (
          <ApiErrorNotice
            error={error}
            endpoint="GET /api/predictions  (optionally GET /api/spectrum)"
            purpose="Provides the stored prediction records used for the availability trend. A dedicated /api/spectrum endpoint would allow true real-time band occupancy monitoring."
          />
        ) : null}

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
          <KpiCard
            label="Total Predictions"
            value={error || modelInfo.error ? "—" : (isPending || modelInfo.isPending) ? "…" : kpis?.total_predictions ?? 0}
            hint="All-time requests"
          />
          <KpiCard
            label="Available"
            value={error || modelInfo.error ? "—" : (isPending || modelInfo.isPending) ? "…" : kpis?.available_predictions ?? 0}
            tone="available"
          />
          <KpiCard
            label="Occupied"
            value={error || modelInfo.error ? "—" : (isPending || modelInfo.isPending) ? "…" : kpis?.occupied_predictions ?? 0}
            tone="occupied"
          />
          <KpiCard
            label="Average probability"
            value={error || modelInfo.error || !kpis ? "—" : (isPending || modelInfo.isPending) ? "…" : formatProbability(kpis.avg_probability)}
            tone="primary"
          />
          <KpiCard
            label="OOD Warnings"
            value={error || modelInfo.error ? "—" : (isPending || modelInfo.isPending) ? "…" : kpis?.ood_count ?? 0}
            tone="destructive"
            hint="Anomalous inputs"
          />
        </div>

        <Panel
          title="Availability trend"
          subtitle="Predictions grouped per hour using each record's timestamp (UTC)."
          action={
            <Button variant="outline" size="sm" onClick={() => void refetch()} disabled={isFetching}>
              <RefreshCw className={isFetching ? "size-3 animate-spin" : "size-3"} />
              Refresh
            </Button>
          }
        >
          {isPending ? (
            <LoadingRow />
          ) : error || trend.length === 0 ? (
            <EmptyRow label="No timestamped records available from the backend." />
          ) : (
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trend}>
                  <CartesianGrid stroke="var(--color-border)" vertical={false} />
                  <XAxis
                    dataKey="hour"
                    tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                    stroke="var(--color-border)"
                  />
                  <YAxis
                    allowDecimals={false}
                    tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                    stroke="var(--color-border)"
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--color-popover)",
                      border: "1px solid var(--color-border)",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="available"
                    stroke="var(--color-available)"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="occupied"
                    stroke="var(--color-occupied)"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>

        <Panel title="Latest activity" subtitle="Most recent stored predictions.">
          {isPending ? (
            <LoadingRow />
          ) : error || records.length === 0 ? (
            <EmptyRow label="No prediction records returned by the backend." />
          ) : (
            <PredictionsTable records={records.slice(0, 12)} />
          )}
        </Panel>
      </div>
    </AppShell>
  );
}
