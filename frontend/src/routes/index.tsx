import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowUpRight } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AppShell } from "@/components/layout/AppShell";
import { ApiErrorNotice, EmptyRow, LoadingRow } from "@/components/wire/ApiStateNotice";
import { KpiCard, Panel } from "@/components/wire/Panel";
import { PredictionsTable } from "@/components/wire/PredictionsTable";
import { deriveStats, usePredictions } from "@/hooks/use-wire-watcher";
import type { PredictionRecord } from "@/types/wire-watcher";
import { formatProbability } from "@/utils/format";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Wire Watcher — Spectrum Intelligence Dashboard" },
      {
        name: "description",
        content:
          "ML-based wireless spectrum availability prediction dashboard powered by a Random Forest model served over a Flask REST API.",
      },
      { property: "og:title", content: "Wire Watcher — Spectrum Intelligence Dashboard" },
      {
        property: "og:description",
        content:
          "Monitor spectrum availability predictions, model confidence and system health for the Wire Watcher ML pipeline.",
      },
    ],
  }),
  component: DashboardPage,
});

interface BandBucket {
  band: string;
  available: number;
  occupied: number;
}

function bucketByBand(records: PredictionRecord[]): BandBucket[] {
  const buckets = new Map<string, BandBucket>();
  for (const record of records) {
    const start = record.start_frequency_mhz;
    if (start === null) continue;
    const floor = Math.floor(start / 100) * 100;
    const key = `${floor}–${floor + 100}`;
    const bucket = buckets.get(key) ?? { band: key, available: 0, occupied: 0 };
    if (record.available === true) bucket.available += 1;
    else if (record.available === false) bucket.occupied += 1;
    buckets.set(key, bucket);
  }
  return [...buckets.values()].sort(
    (a, b) => Number(a.band.split("–")[0]) - Number(b.band.split("–")[0]),
  );
}

function DashboardPage() {
  const { data, isPending, error } = usePredictions();
  const records = data ?? [];
  const stats = deriveStats(records);
  const bands = bucketByBand(records);
  const recent = records.slice(0, 8);

  return (
    <AppShell
      title="Spectrum Intelligence Dashboard"
      description="Live view of predictions stored by the Flask + Random Forest + MySQL pipeline."
    >
      <div className="space-y-6">
        {error ? (
          <ApiErrorNotice
            error={error}
            endpoint="GET /api/predictions  (and optionally GET /api/statistics)"
            purpose="Returns stored prediction records used for the KPI cards, recent activity table and spectrum charts. A dedicated /api/statistics endpoint would let the dashboard read pre-aggregated totals instead of computing them from records."
          />
        ) : null}

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <KpiCard
            label="Total predictions"
            value={error ? "—" : isPending ? "…" : stats.total}
            hint="Records returned by GET /api/predictions"
          />
          <KpiCard
            label="Available channels"
            value={error ? "—" : isPending ? "…" : stats.available}
            tone="available"
            hint="Records with available = true"
          />
          <KpiCard
            label="Occupied channels"
            value={error ? "—" : isPending ? "…" : stats.occupied}
            tone="occupied"
            hint="Records with available = false"
          />
          <KpiCard
            label="Average probability"
            value={error || isPending ? "—" : formatProbability(stats.averageProbability)}
            tone="primary"
            hint="Mean model probability across returned records"
          />
        </div>

        {!error && stats.unknownLabel > 0 ? (
          <p className="text-xs text-warning">
            {stats.unknownLabel} record(s) had no readable availability field and are excluded from
            the available/occupied counts.
          </p>
        ) : null}

        <div className="grid gap-6 xl:grid-cols-[3fr_2fr]">
          <Panel
            title="Recent predictions"
            subtitle="Most recent records as returned by the backend."
            action={
              <Link
                to="/history"
                className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
              >
                Full history <ArrowUpRight className="size-3" />
              </Link>
            }
          >
            {isPending ? (
              <LoadingRow />
            ) : error ? (
              <EmptyRow label="No records available — the endpoint above is required." />
            ) : recent.length === 0 ? (
              <EmptyRow label="The backend returned no prediction records yet." />
            ) : (
              <PredictionsTable records={recent} />
            )}
          </Panel>

          <Panel
            title="Spectrum overview"
            subtitle="Availability by 100 MHz band, computed from returned records."
          >
            {isPending ? (
              <LoadingRow />
            ) : error || bands.length === 0 ? (
              <EmptyRow label="No frequency data available from the backend." />
            ) : (
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={bands} barGap={2}>
                    <CartesianGrid stroke="var(--color-border)" vertical={false} />
                    <XAxis
                      dataKey="band"
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
                    <Bar dataKey="available" stackId="a" fill="var(--color-available)" radius={[0, 0, 0, 0]}>
                      {bands.map((b) => (
                        <Cell key={`a-${b.band}`} />
                      ))}
                    </Bar>
                    <Bar dataKey="occupied" stackId="a" fill="var(--color-occupied)" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </Panel>
        </div>
      </div>
    </AppShell>
  );
}
