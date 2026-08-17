export const DASH = "—";

export function formatNumber(value: number | null | undefined, digits = 0, suffix = ""): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `${value.toFixed(digits)}${suffix}`;
}

export function formatProbability(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `${(value * 100).toFixed(1)}%`;
}

export function formatFrequencyRange(start: number | null, end: number | null): string {
  if (start === null && end === null) return DASH;
  if (start !== null && end !== null) return `${start} – ${end} MHz`;
  return `${start ?? end} MHz`;
}

export function formatLocation(city: string | null, state: string | null): string {
  const parts = [city, state].filter((p): p is string => Boolean(p));
  return parts.length ? parts.join(", ") : DASH;
}

export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return DASH;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export const DAY_LABELS = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
];
