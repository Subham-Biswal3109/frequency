import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

import { AppShell } from "@/components/layout/AppShell";
import { ApiErrorNotice } from "@/components/wire/ApiStateNotice";
import { AllocationResultCard } from "@/components/wire/simulation/AllocationResultCard";
import { AllocationSteps } from "@/components/wire/simulation/AllocationSteps";
import { CandidateRankingTable } from "@/components/wire/simulation/CandidateRankingTable";
import { ChannelOccupancyBar } from "@/components/wire/simulation/ChannelOccupancyBar";
import { ChannelSpectrumChart } from "@/components/wire/simulation/ChannelSpectrumChart";
import { ChannelTable } from "@/components/wire/simulation/ChannelTable";
import { MultiUserResults } from "@/components/wire/simulation/MultiUserResults";
import { ResourceUtilizationBar } from "@/components/wire/simulation/ResourceUtilizationBar";
import { SimulationConfigForm } from "@/components/wire/simulation/SimulationConfigForm";
import { SnrSensitivityChart } from "@/components/wire/simulation/SnrSensitivityChart";
import { useRunSimulation, useSnrSweep } from "@/hooks/use-wire-watcher";
import type { SimulationRunRequest, SimulationRunResponse, SnrSweepPoint } from "@/types/wire-watcher";

export const Route = createFileRoute("/simulation")({
  head: () => ({
    meta: [
      { title: "Spectrum Simulation — Wire Watcher" },
      {
        name: "description",
        content:
          "Simulated spectrum sensing and availability-based channel allocation — an educational demonstration built on Wire Watcher's existing RF pipeline and ML model.",
      },
      { property: "og:title", content: "Spectrum Simulation — Wire Watcher" },
      {
        property: "og:description",
        content:
          "Configure a simulated RF environment, watch spectrum sensing and ML-assisted availability estimation run, and see a transparent channel allocation decision.",
      },
    ],
  }),
  component: SimulationPage,
});

function SimulationPage() {
  const runSimulation = useRunSimulation();
  const snrSweep = useSnrSweep();
  const [result, setResult] = useState<SimulationRunResponse | null>(null);
  const [snrPoints, setSnrPoints] = useState<SnrSweepPoint[] | null>(null);

  const handleRun = (input: SimulationRunRequest) => {
    runSimulation.mutate(input, {
      onSuccess: (data) => setResult(data),
    });
  };

  const handleSnrSweep = () => {
    snrSweep.mutate(
      {},
      {
        onSuccess: (data) => setSnrPoints(data.points),
      },
    );
  };

  return (
    <AppShell
      title="Spectrum Simulation"
      description="Simulated spectrum sensing and availability-based channel allocation — an engineering demonstration built on Wire Watcher's existing RF pipeline and Random Forest model."
    >
      <div className="space-y-6">
        <p className="rounded-md border border-warning/30 bg-warning/8 px-3 py-2 text-xs text-warning">
          This module does not claim to represent the exact spectrum-allocation mechanism used by TRAI,
          DoT, or telecom operators. It demonstrates the engineering concept of simulated spectrum
          sensing and availability-based channel allocation using synthetic RF conditions. Results do
          not represent live spectrum measurements or official spectrum allocation.
        </p>

        <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
          <SimulationConfigForm onSubmit={handleRun} pending={runSimulation.isPending} />

          <div className="space-y-6">
            {runSimulation.isError ? (
              <ApiErrorNotice
                error={runSimulation.error}
                endpoint="POST /api/simulation/run"
                purpose="Generates a simulated RF environment, runs RF sensing and ML availability estimation, and allocates spectrum."
              />
            ) : null}

            {!result && !runSimulation.isPending ? (
              <div className="panel flex h-64 items-center justify-center px-4 text-center text-sm text-muted-foreground">
                Configure a frequency range and click "Generate Spectrum &amp; Allocate" to run the
                simulation.
              </div>
            ) : null}

            {result ? (
              <>
                <AllocationSteps modelLoaded={result.model_loaded} />

                <ChannelSpectrumChart
                  frequencies={result.spectrum_data.frequencies}
                  powers={result.spectrum_data.power_dbm}
                  channels={result.allocation?.final_channels ?? result.multi_user_allocation?.final_channels ?? result.channels}
                  noiseFloor={result.noise_floor_dbm}
                />

                <ChannelOccupancyBar
                  channels={result.allocation?.final_channels ?? result.multi_user_allocation?.final_channels ?? result.channels}
                />

                {result.mode === "multi_user" && result.multi_user_allocation ? (
                  <>
                    <MultiUserResults allocation={result.multi_user_allocation} />
                    <ResourceUtilizationBar
                      before={result.resource_utilization_before}
                      after={result.resource_utilization_after}
                    />
                  </>
                ) : result.allocation ? (
                  <>
                    <AllocationResultCard
                      success={result.allocation.success}
                      selected={result.allocation.selected}
                      message={result.allocation.message}
                      requestedBandwidth={result.allocation.requested_bandwidth_mhz}
                    />
                    <CandidateRankingTable
                      candidates={result.allocation.top_candidates}
                      selectedRank={result.allocation.selected?.rank}
                    />
                    <ResourceUtilizationBar
                      before={result.resource_utilization_before}
                      after={result.resource_utilization_after}
                    />
                  </>
                ) : null}

                <ChannelTable channels={result.allocation?.final_channels ?? result.multi_user_allocation?.final_channels ?? result.channels} />
              </>
            ) : null}

            <SnrSensitivityChart points={snrPoints} onRun={handleSnrSweep} pending={snrSweep.isPending} />
          </div>
        </div>
      </div>
    </AppShell>
  );
}
