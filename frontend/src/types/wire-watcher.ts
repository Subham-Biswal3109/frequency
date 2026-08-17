/**
 * Types mirroring the EXISTING Flask API contract for Wire Watcher.
 * Field names come from the running backend (POST /api/predict) and must not be renamed.
 */

export interface AnalyzeSpectrumResponse {
  data_source: string;
  frequency_range: { start_mhz: number; end_mhz: number };
  noise_floor_dbm: number;
  detected_signals: Array<{
    frequency_mhz: number;
    power_dbm: number;
    bandwidth_mhz: number;
    snr_db: number;
  }>;
  occupied_regions: Array<{ start_mhz: number; end_mhz: number }>;
  available_regions: Array<{ start_mhz: number; end_mhz: number }>;
  spectrum_data: {
    frequencies: number[];
    power_dbm: number[];
  };
  extracted_features: PredictRequest;
}

export interface PredictRequest {
  start_frequency_mhz: number;
  end_frequency_mhz: number;
  bandwidth_mhz: number;
  hour_of_day: number;
  day_of_week: number;
  signal_power_dbm: number;
  noise_floor_dbm: number;
  snr_db: number;
  state: string;
  city: string;
  service_type: string;
  region?: string;
  latitude?: number;
  longitude?: number;
}

export interface PredictResponse {
  prediction: number;
  available: boolean;
  probability: number;
  confidence: string;
  data_source: string;
  threshold?: number;
  important_features: string[];
  features_used: Record<string, string | number>;
  ood_warning?: boolean;
  warning?: string;
}

/**
 * Normalized view of a stored prediction record coming from GET /api/predictions.
 * The backend row shape is read tolerantly: any field the backend does not
 * return stays `null` and is rendered as "—" (never faked).
 */
export interface PredictionRecord {
  id: string;
  start_frequency_mhz: number | null;
  end_frequency_mhz: number | null;
  bandwidth_mhz: number | null;
  city: string | null;
  state: string | null;
  service_type: string | null;
  available: boolean | null;
  probability: number | null;
  timestamp: string | null;
  signal_power_dbm?: number | null;
  noise_floor_dbm?: number | null;
  snr_db?: number | null;
  data_source?: string;
  ood_status?: boolean;
  raw: Record<string, unknown>;
}

/** Shape of GET /api/health once the backend exposes it. */
export interface HealthResponse {
  status?: string;
  api?: string;
  model?: string;
  model_loaded?: boolean;
  database?: string;
  database_connected?: boolean;
  [key: string]: unknown;
}

export interface ModelInfoResponse {
  algorithm?: string;
  model_version?: string;
  training_date?: string;
  dataset_name?: string;
  dataset_type?: string;
  training_samples?: number;
  real_rf_validation?: boolean;
  best_threshold?: number;
  data_source?: string;
  features?: string[];
  limitations?: string[];
  kpis?: {
    total_predictions: number;
    available_predictions: number;
    occupied_predictions: number;
    avg_probability: number;
    ood_count: number;
  };
  [key: string]: unknown;
}

export type ApiErrorKind =
  | "network"
  | "timeout"
  | "validation"
  | "server"
  | "malformed"
  | "not_implemented";
