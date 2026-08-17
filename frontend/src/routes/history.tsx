import { createFileRoute } from "@tanstack/react-router";
import { ArrowDownUp, RefreshCw, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AppShell } from "@/components/layout/AppShell";
import { ApiErrorNotice, EmptyRow, LoadingRow } from "@/components/wire/ApiStateNotice";
import { Panel } from "@/components/wire/Panel";
import { PredictionsTable } from "@/components/wire/PredictionsTable";
import { usePredictions } from "@/hooks/use-wire-watcher";
import type { PredictionRecord } from "@/types/wire-watcher";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Prediction History — Wire Watcher" },
      {
        name: "description",
        content:
          "Search, filter and sort the spectrum availability predictions stored in the Wire Watcher MySQL database.",
      },
      { property: "og:title", content: "Prediction History — Wire Watcher" },
      {
        property: "og:description",
        content:
          "Browse stored Wire Watcher predictions with filters for availability, location, frequency range and date.",
      },
    ],
  }),
  component: HistoryPage,
});

type SortKey = "timestamp" | "start_frequency_mhz" | "probability";

function HistoryPage() {
  const { data, isPending, error, refetch, isFetching } = usePredictions();
  const records = data ?? [];

  const [search, setSearch] = useState("");
  const [availability, setAvailability] = useState<"all" | "available" | "occupied">("all");
  const [location, setLocation] = useState("");
  const [freqMin, setFreqMin] = useState("");
  const [freqMax, setFreqMax] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("timestamp");
  const [sortDesc, setSortDesc] = useState(true);

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase();
    const loc = location.trim().toLowerCase();
    const min = freqMin === "" ? null : Number(freqMin);
    const max = freqMax === "" ? null : Number(freqMax);
    const from = dateFrom === "" ? null : new Date(dateFrom).getTime();
    const to = dateTo === "" ? null : new Date(dateTo).getTime() + 86_400_000;

    const rows = records.filter((r) => {
      if (availability === "available" && r.available !== true) return false;
      if (availability === "occupied" && r.available !== false) return false;

      if (loc) {
        const haystack = `${r.city ?? ""} ${r.state ?? ""}`.toLowerCase();
        if (!haystack.includes(loc)) return false;
      }
      if (min !== null && (r.start_frequency_mhz === null || r.start_frequency_mhz < min)) return false;
      if (max !== null) {
        const end = r.end_frequency_mhz ?? r.start_frequency_mhz;
        if (end === null || end > max) return false;
      }
      if (from !== null || to !== null) {
        const time = r.timestamp ? new Date(r.timestamp).getTime() : NaN;
        if (Number.isNaN(time)) return false;
        if (from !== null && time < from) return false;
        if (to !== null && time > to) return false;
      }
      if (term) {
        const haystack = JSON.stringify(r.raw).toLowerCase();
        if (!haystack.includes(term)) return false;
      }
      return true;
    });

    const value = (r: PredictionRecord): number => {
      if (sortKey === "timestamp") return r.timestamp ? new Date(r.timestamp).getTime() || 0 : 0;
      return r[sortKey] ?? -Infinity;
    };
    return rows.sort((a, b) => (sortDesc ? value(b) - value(a) : value(a) - value(b)));
  }, [records, search, availability, location, freqMin, freqMax, dateFrom, dateTo, sortKey, sortDesc]);

  return (
    <AppShell
      title="Prediction History"
      description="Records read from MySQL through GET /api/predictions. Fields the backend does not return are shown as —."
    >
      <div className="space-y-6">
        {error ? (
          <ApiErrorNotice
            error={error}
            endpoint="GET /api/predictions"
            purpose="Returns stored prediction records (frequency range, bandwidth, city/state, availability, probability, timestamp) for the history table."
          />
        ) : null}

        <Panel title="Filters" subtitle="All filtering happens on the records returned by the backend.">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="space-y-1.5">
              <Label className="label-caps">Search</Label>
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  className="pl-9"
                  placeholder="Any field…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label className="label-caps">Prediction</Label>
              <select
                className="h-9 w-full rounded-md border border-input bg-surface px-3 text-sm"
                value={availability}
                onChange={(e) => setAvailability(e.target.value as typeof availability)}
              >
                <option value="all">All</option>
                <option value="available">Available</option>
                <option value="occupied">Occupied</option>
              </select>
            </div>
            <div className="space-y-1.5">
              <Label className="label-caps">Location</Label>
              <Input
                placeholder="City or state"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="label-caps">Frequency range (MHz)</Label>
              <div className="flex gap-2">
                <Input
                  type="number"
                  placeholder="min"
                  value={freqMin}
                  onChange={(e) => setFreqMin(e.target.value)}
                />
                <Input
                  type="number"
                  placeholder="max"
                  value={freqMax}
                  onChange={(e) => setFreqMax(e.target.value)}
                />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label className="label-caps">Date from</Label>
              <Input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label className="label-caps">Date to</Label>
              <Input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label className="label-caps">Sort by</Label>
              <select
                className="h-9 w-full rounded-md border border-input bg-surface px-3 text-sm"
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value as SortKey)}
              >
                <option value="timestamp">Timestamp</option>
                <option value="start_frequency_mhz">Start frequency</option>
                <option value="probability">Probability</option>
              </select>
            </div>
            <div className="flex items-end gap-2">
              <Button
                type="button"
                variant="secondary"
                className="gap-2"
                onClick={() => setSortDesc((v) => !v)}
              >
                <ArrowDownUp className="size-4" />
                {sortDesc ? "Descending" : "Ascending"}
              </Button>
              <Button type="button" variant="outline" className="gap-2" onClick={() => refetch()}>
                <RefreshCw className={isFetching ? "size-4 animate-spin" : "size-4"} />
                Reload
              </Button>
            </div>
          </div>
        </Panel>

        <Panel
          title="Stored predictions"
          subtitle={
            error
              ? "Unavailable"
              : `${filtered.length} of ${records.length} record(s) shown`
          }
        >
          {isPending ? (
            <LoadingRow />
          ) : error ? (
            <EmptyRow label="No records — the backend endpoint above is required." />
          ) : filtered.length === 0 ? (
            <EmptyRow
              label={
                records.length === 0
                  ? "The backend returned no prediction records."
                  : "No records match the current filters."
              }
            />
          ) : (
            <PredictionsTable records={filtered} />
          )}
        </Panel>
      </div>
    </AppShell>
  );
}
