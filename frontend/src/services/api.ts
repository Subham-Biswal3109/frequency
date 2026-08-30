/**
 * Centralized Wire Watcher API service.
 * Every request to the existing Flask backend goes through this module.
 * Base URL comes from VITE_API_BASE_URL (dev: http://localhost:5000).
 */

import type {
  ApiErrorKind,
  HealthResponse,
  PredictRequest,
  PredictResponse,
  PredictionRecord,
  AnalyzeSpectrumResponse,
  SimulationRunRequest,
  SimulationRunResponse,
  SnrSweepResponse,
  JammingModelInfoResponse,
  JammingSamplesResponse,
  JammingPredictResponse,
} from "@/types/wire-watcher";

export const API_BASE_URL = (
  import.meta.env['VITE_API_BASE_URL'] ?? "http://localhost:5000"
).replace(/\/+$/, "");

const DEFAULT_TIMEOUT_MS = 15000;

export class ApiError extends Error {
  kind: ApiErrorKind;
  status?: number | undefined;
  details?: unknown;

  constructor(
    kind: ApiErrorKind,
    message: string,
    options?: { status?: number; details?: unknown },
  ) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = options?.status;
    this.details = options?.details;
  }
}

export function describeError(error: unknown): { title: string; message: string; details?: string[] | undefined } {
  if (error instanceof ApiError) {
    const details = Array.isArray(error.details)
      ? error.details.map((d) => (typeof d === "string" ? d : JSON.stringify(d)))
      : typeof error.details === "string"
        ? [error.details]
        : undefined;
    const titles: Record<ApiErrorKind, string> = {
      network: "Backend unavailable",
      timeout: "Request timed out",
      validation: "Invalid input",
      server: "Backend error",
      malformed: "Malformed response",
      not_implemented: "Endpoint not implemented",
    };
    return { title: titles[error.kind], message: error.message, details };
  }
  return {
    title: "Unexpected error",
    message: error instanceof Error ? error.message : String(error),
  };
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    });
  } catch (error) {
    if (controller.signal.aborted) {
      throw new ApiError(
        "timeout",
        `No response from ${API_BASE_URL}${path} within ${timeoutMs / 1000}s.`,
      );
    }
    throw new ApiError(
      "network",
      `Could not reach the Flask API at ${API_BASE_URL}. Is it running?`,
      { details: error instanceof Error ? error.message : undefined },
    );
  } finally {
    clearTimeout(timer);
  }

  const text = await response.text();
  let body: unknown = undefined;
  if (text.length > 0) {
    try {
      body = JSON.parse(text);
    } catch {
      if (response.ok) {
        throw new ApiError("malformed", `Response from ${path} was not valid JSON.`, {
          status: response.status,
          details: text.slice(0, 300),
        });
      }
    }
  }

  const payload = (body ?? {}) as { error?: string; details?: unknown; message?: string };

  if (!response.ok) {
    const message = payload.error ?? payload.message ?? text.slice(0, 300) ?? response.statusText;
    if (response.status === 404) {
      throw new ApiError("not_implemented", `${path} returned 404 — the endpoint does not exist on the Flask API.`, {
        status: 404,
      });
    }
    if (response.status === 400 || response.status === 422) {
      throw new ApiError("validation", message || "Invalid input parameters", {
        status: response.status,
        details: payload.details,
      });
    }
    throw new ApiError("server", message || `Request failed with status ${response.status}.`, {
      status: response.status,
      details: payload.details,
    });
  }

  if (body === undefined) {
    throw new ApiError("malformed", `Empty response body from ${path}.`, { status: response.status });
  }

  return body as T;
}

/* ------------------------------- predict -------------------------------- */

export async function predict(input: PredictRequest): Promise<PredictResponse> {
  const body = await request<unknown>("/api/predict", {
    method: "POST",
    body: JSON.stringify(input),
  });

  const candidate = body as Partial<PredictResponse>;
  const valid =
    candidate !== null &&
    typeof candidate === "object" &&
    typeof candidate.available === "boolean" &&
    typeof candidate.probability === "number" &&
    typeof candidate.prediction === "number";

  if (!valid) {
    throw new ApiError(
      "malformed",
      "POST /api/predict responded without the expected keys (prediction, available, probability).",
      { details: JSON.stringify(body).slice(0, 400) },
    );
  }
  return candidate as PredictResponse;
}

/* ------------------------------- spectrum ------------------------------- */

export async function analyzeSpectrum(input: Partial<PredictRequest>): Promise<AnalyzeSpectrumResponse> {
  return request<AnalyzeSpectrumResponse>("/api/spectrum/analyze", {
    method: "POST",
    body: JSON.stringify({
      center_freq_mhz: input.start_frequency_mhz ? input.start_frequency_mhz + ((input.bandwidth_mhz || 10) / 2) : 1800,
      bandwidth_mhz: input.bandwidth_mhz || 10,
      signal_strength_dbm: input.signal_power_dbm || -75,
      noise_floor_dbm: input.noise_floor_dbm || -100,
      state: input.state,
      city: input.city,
      service_type: input.service_type
    }),
  });
}

/* ----------------------------- predictions ------------------------------ */

function pick(row: Record<string, unknown>, keys: string[]): unknown {
  for (const key of keys) {
    if (row[key] !== undefined && row[key] !== null) return row[key];
  }
  return undefined;
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))) {
    return Number(value);
  }
  return null;
}

function asString(value: unknown): string | null {
  if (typeof value === "string" && value.trim() !== "") return value;
  if (typeof value === "number") return String(value);
  return null;
}

function asBoolean(value: unknown): boolean | null {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value === 1 ? true : value === 0 ? false : null;
  if (typeof value === "string") {
    const v = value.trim().toLowerCase();
    if (["1", "true", "available", "yes"].includes(v)) return true;
    if (["0", "false", "occupied", "unavailable", "not available", "no"].includes(v)) return false;
  }
  return null;
}

function normalizeRecord(row: Record<string, unknown>, index: number): PredictionRecord {
  const probabilityRaw = asNumber(pick(row, ["probability", "confidence", "prob", "probability_available"]));
  const probability =
    probabilityRaw === null ? null : probabilityRaw > 1 ? probabilityRaw / 100 : probabilityRaw;

  return {
    id: asString(pick(row, ["id", "prediction_id", "_id"])) ?? `row-${index}`,
    start_frequency_mhz: asNumber(pick(row, ["start_frequency_mhz", "start_frequency", "frequency_mhz", "frequency"])),
    end_frequency_mhz: asNumber(pick(row, ["end_frequency_mhz", "end_frequency"])),
    bandwidth_mhz: asNumber(pick(row, ["bandwidth_mhz", "bandwidth"])),
    city: asString(pick(row, ["city", "location"])),
    state: asString(pick(row, ["state"])),
    service_type: asString(pick(row, ["service_type", "service"])),
    available: asBoolean(pick(row, ["available", "prediction", "is_available", "prediction_label"])),
    probability,
    timestamp: asString(pick(row, ["timestamp", "created_at", "predicted_at", "prediction_time", "time"])),
    raw: row,
  };
}

export async function getPredictions(): Promise<PredictionRecord[]> {
  const body = await request<unknown>("/api/predictions", { method: "GET" });

  const rows: unknown =
    Array.isArray(body)
      ? body
      : typeof body === "object" && body !== null
        ? ((body as Record<string, unknown>)["predictions"] ??
          (body as Record<string, unknown>)["data"] ??
          (body as Record<string, unknown>)["records"] ??
          (body as Record<string, unknown>)["results"])
        : undefined;

  if (!Array.isArray(rows)) {
    throw new ApiError(
      "malformed",
      "GET /api/predictions did not return an array of records (checked the root array and predictions/data/records/results keys).",
      { details: JSON.stringify(body).slice(0, 400) },
    );
  }

  return rows
    .filter((row): row is Record<string, unknown> => typeof row === "object" && row !== null)
    .map(normalizeRecord);
}

/* -------------------------------- health -------------------------------- */

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", { method: "GET" }, 8000);
}

export async function getModelInfo(): Promise<import("@/types/wire-watcher").ModelInfoResponse> {
  return request<import("@/types/wire-watcher").ModelInfoResponse>("/api/model-info", { method: "GET" }, 8000);
}

/* ---------------------------- spectrum simulation ------------------------
 * Separate module: POST /api/simulation/run, POST /api/simulation/snr-sweep.
 * Does not touch the /api/predict, /api/spectrum/analyze code paths above.
 * ------------------------------------------------------------------------- */

export async function runSimulation(input: SimulationRunRequest): Promise<SimulationRunResponse> {
  return request<SimulationRunResponse>(
    "/api/simulation/run",
    { method: "POST", body: JSON.stringify(input) },
    30000, // simulations with many channels can take longer than the default 15s
  );
}

export async function runSnrSweep(input: {
  signal_power_dbm?: number;
  start_frequency_mhz?: number;
  end_frequency_mhz?: number;
  bandwidth_mhz?: number;
  state?: string;
  city?: string;
  service_type?: string;
}): Promise<SnrSweepResponse> {
  return request<SnrSweepResponse>("/api/simulation/snr-sweep", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

/* ---------------------------- jamming detector ----------------------------
 * A SEPARATE model/task (benign vs malicious RF activity), not spectrum
 * availability. Does not touch the /api/predict or /api/simulation/* code
 * paths above.
 * ------------------------------------------------------------------------- */

export async function getJammingModelInfo(): Promise<JammingModelInfoResponse> {
  return request<JammingModelInfoResponse>("/api/jamming/model-info", { method: "GET" }, 8000);
}

export async function getJammingSamples(): Promise<JammingSamplesResponse> {
  return request<JammingSamplesResponse>("/api/jamming/samples", { method: "GET" }, 8000);
}

export async function predictJamming(sampleId: string): Promise<JammingPredictResponse> {
  return request<JammingPredictResponse>("/api/jamming/predict", {
    method: "POST",
    body: JSON.stringify({ sample_id: sampleId }),
  });
}
