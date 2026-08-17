import { useMutation, useQuery } from "@tanstack/react-query";

import { getHealth, getPredictions, predict, analyzeSpectrum } from "@/services/api";
import { ApiError } from "@/services/api";
import type { PredictRequest, PredictionRecord, AnalyzeSpectrumResponse } from "@/types/wire-watcher";

const noRetryOnClientError = (failureCount: number, error: unknown) => {
  if (error instanceof ApiError && ["not_implemented", "validation", "malformed"].includes(error.kind)) {
    return false;
  }
  return failureCount < 1;
};

/** GET /api/predictions — exists on the backend. Client-side only (backend is on localhost). */
export function usePredictions() {
  return useQuery({
    queryKey: ["predictions"],
    queryFn: getPredictions,
    retry: noRetryOnClientError,
    staleTime: 15_000,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    retry: noRetryOnClientError,
    refetchInterval: 30_000,
  });
}

/** GET /api/model-info */
export function useModelInfo() {
  return useQuery({
    queryKey: ["model-info"],
    queryFn: async () => {
      const { getModelInfo } = await import("@/services/api");
      return getModelInfo();
    },
    retry: noRetryOnClientError,
    staleTime: 60_000,
  });
}

export function usePredict() {
  return useMutation({
    mutationFn: (input: PredictRequest) => predict(input),
  });
}

export function useAnalyzeSpectrum() {
  return useMutation({
    mutationFn: (input: Partial<PredictRequest>) => analyzeSpectrum(input),
  });
}

export interface DerivedStats {
  total: number;
  available: number;
  occupied: number;
  unknownLabel: number;
  averageProbability: number | null;
}

/** Aggregates REAL records returned by GET /api/predictions. No values are invented. */
export function deriveStats(records: PredictionRecord[]): DerivedStats {
  const available = records.filter((r) => r.available === true).length;
  const occupied = records.filter((r) => r.available === false).length;
  const probabilities = records
    .map((r) => r.probability)
    .filter((p): p is number => typeof p === "number");

  return {
    total: records.length,
    available,
    occupied,
    unknownLabel: records.length - available - occupied,
    averageProbability:
      probabilities.length > 0
        ? probabilities.reduce((a, b) => a + b, 0) / probabilities.length
        : null,
  };
}
