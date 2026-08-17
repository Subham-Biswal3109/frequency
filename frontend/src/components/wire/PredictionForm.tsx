import { useEffect } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { Loader2, Radar, Activity } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Panel } from "@/components/wire/Panel";
import type { PredictRequest } from "@/types/wire-watcher";
import { DAY_LABELS } from "@/utils/format";

/** Validation mirrors the Flask endpoint's documented constraints exactly. */
const schema = z.object({
  start_frequency_mhz: z.coerce.number().finite().positive(),
  end_frequency_mhz: z.coerce.number().finite().positive(),
  bandwidth_mhz: z.coerce.number().finite().positive(),
  hour_of_day: z.coerce.number().int().min(0).max(23),
  day_of_week: z.coerce.number().int().min(0).max(6),
  signal_power_dbm: z.coerce.number().finite(),
  noise_floor_dbm: z.coerce.number().finite(),
  snr_db: z.coerce.number().finite(),
  state: z.string().trim().min(1, "Required"),
  city: z.string().trim().min(1, "Required"),
  service_type: z.string().trim().min(1, "Required"),
  region: z.string().trim().optional(),
  latitude: z.union([z.coerce.number().min(-90).max(90), z.literal("")]).optional(),
  longitude: z.union([z.coerce.number().min(-180).max(180), z.literal("")]).optional(),
});

type FormValues = z.input<typeof schema>;

function Field({
  label,
  hint,
  error,
  children,
}: {
  label: string;
  hint?: string;
  error?: string | undefined;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="label-caps">{label}</Label>
      {children}
      {error ? (
        <p className="text-xs text-destructive">{error}</p>
      ) : hint ? (
        <p className="text-xs text-muted-foreground">{hint}</p>
      ) : null}
    </div>
  );
}

export function PredictionForm({
  onSubmit,
  onAnalyze,
  pending,
  analyzing,
}: {
  onSubmit: (input: PredictRequest) => void;
  onAnalyze?: (input: Partial<PredictRequest>) => void;
  pending: boolean;
  analyzing?: boolean;
}) {
  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      start_frequency_mhz: 1800,
      end_frequency_mhz: 1810,
      bandwidth_mhz: 10,
      hour_of_day: new Date().getHours(),
      day_of_week: (new Date().getDay() + 6) % 7,
      signal_power_dbm: -85.5,
      noise_floor_dbm: -105,
      snr_db: 19.5,
      state: "Maharashtra",
      city: "Mumbai",
      service_type: "4G LTE",
      region: "",
      latitude: "",
      longitude: "",
    },
  });

  const signal = watch("signal_power_dbm");
  const noise = watch("noise_floor_dbm");

  // Auto-calculate SNR when signal or noise changes
  useEffect(() => {
    if (!isNaN(signal) && !isNaN(noise)) {
      setValue("snr_db", Number((signal - noise).toFixed(2)));
    }
  }, [signal, noise, setValue]);

  const submit = handleSubmit((values) => {
    const parsed = schema.parse(values);
    const payload: PredictRequest = {
      start_frequency_mhz: parsed.start_frequency_mhz,
      end_frequency_mhz: parsed.end_frequency_mhz,
      bandwidth_mhz: parsed.bandwidth_mhz,
      hour_of_day: parsed.hour_of_day,
      day_of_week: parsed.day_of_week,
      signal_power_dbm: parsed.signal_power_dbm,
      noise_floor_dbm: parsed.noise_floor_dbm,
      snr_db: parsed.snr_db,
      state: parsed.state,
      city: parsed.city,
      service_type: parsed.service_type,
    };
    if (parsed.region) payload.region = parsed.region;
    if (typeof parsed.latitude === "number") payload.latitude = parsed.latitude;
    if (typeof parsed.longitude === "number") payload.longitude = parsed.longitude;
    onSubmit(payload);
  });

  const applyPreset = (preset: Partial<FormValues>) => {
    Object.entries(preset).forEach(([key, value]) => {
      setValue(key as keyof FormValues, value);
    });
  };

  return (
    <form onSubmit={submit} className="space-y-5">
      <Panel title="Real-World RF Scenarios" subtitle="Populate the form with common realistic scenarios.">
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="secondary" size="sm" onClick={() => applyPreset({ start_frequency_mhz: 1800, end_frequency_mhz: 1810, bandwidth_mhz: 10, signal_power_dbm: -103, noise_floor_dbm: -100 })}>Noise Dominated</Button>
          <Button type="button" variant="secondary" size="sm" onClick={() => applyPreset({ start_frequency_mhz: 2100, end_frequency_mhz: 2110, bandwidth_mhz: 10, signal_power_dbm: -96, noise_floor_dbm: -103 })}>Weak Signal</Button>
          <Button type="button" variant="secondary" size="sm" onClick={() => applyPreset({ start_frequency_mhz: 2100, end_frequency_mhz: 2110, bandwidth_mhz: 10, signal_power_dbm: -80, noise_floor_dbm: -100 })}>Intermediate Signal</Button>
          <Button type="button" variant="secondary" size="sm" onClick={() => applyPreset({ start_frequency_mhz: 2300, end_frequency_mhz: 2320, bandwidth_mhz: 20, signal_power_dbm: -70, noise_floor_dbm: -103 })}>Strong Signal</Button>
          <Button type="button" variant="secondary" size="sm" onClick={() => applyPreset({ start_frequency_mhz: 900, end_frequency_mhz: 905, bandwidth_mhz: 5, signal_power_dbm: -101, noise_floor_dbm: -99 })}>Below Noise Floor</Button>
        </div>
      </Panel>
      <Panel title="Frequency band" subtitle="Values are sent to the model exactly as entered.">
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Start frequency (MHz)" error={errors.start_frequency_mhz?.message}>
            <Input type="number" step="any" {...register("start_frequency_mhz")} />
          </Field>
          <Field label="End frequency (MHz)" error={errors.end_frequency_mhz?.message}>
            <Input type="number" step="any" {...register("end_frequency_mhz")} />
          </Field>
          <Field label="Bandwidth (MHz)" error={errors.bandwidth_mhz?.message}>
            <Input type="number" step="any" {...register("bandwidth_mhz")} />
          </Field>
        </div>
      </Panel>

      <Panel title="Signal measurements">
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Signal power (dBm)" error={errors.signal_power_dbm?.message}>
            <Input type="number" step="any" {...register("signal_power_dbm")} />
          </Field>
          <Field label="Noise floor (dBm)" error={errors.noise_floor_dbm?.message}>
            <Input type="number" step="any" {...register("noise_floor_dbm")} />
          </Field>
          <Field label="SNR (dB)" error={errors.snr_db?.message}>
            <Input type="number" step="any" {...register("snr_db")} />
          </Field>
        </div>
      </Panel>

      <Panel title="Time context">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Hour of day (0–23)" error={errors.hour_of_day?.message}>
            <Input type="number" min={0} max={23} step={1} {...register("hour_of_day")} />
          </Field>
          <Field
            label="Day of week"
            hint="Backend encoding: 0 = Monday … 6 = Sunday"
            error={errors.day_of_week?.message}
          >
            <select
              className="h-9 w-full rounded-md border border-input bg-surface px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
              {...register("day_of_week")}
            >
              {DAY_LABELS.map((day, index) => (
                <option key={day} value={index}>
                  {index} — {day}
                </option>
              ))}
            </select>
          </Field>
        </div>
      </Panel>

      <Panel title="Location & service">
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="State" error={errors.state?.message}>
            <Input placeholder="Karnataka" {...register("state")} />
          </Field>
          <Field label="City" error={errors.city?.message}>
            <Input placeholder="Bangalore" {...register("city")} />
          </Field>
          <Field label="Service type" error={errors.service_type?.message}>
            <Input placeholder="4G LTE" {...register("service_type")} />
          </Field>
        </div>
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          <Field label="Region (optional)" error={errors.region?.message}>
            <Input placeholder="South India" {...register("region")} />
          </Field>
          <Field label="Latitude (optional)" error={errors.latitude?.message}>
            <Input type="number" step="any" placeholder="12.9716" {...register("latitude")} />
          </Field>
          <Field label="Longitude (optional)" error={errors.longitude?.message}>
            <Input type="number" step="any" placeholder="77.5946" {...register("longitude")} />
          </Field>
        </div>
      </Panel>

      <div className="flex flex-wrap items-center gap-3">
        {onAnalyze && (
          <Button type="button" variant="secondary" disabled={analyzing} onClick={handleSubmit(onAnalyze)} className="gap-2">
            {analyzing ? <Loader2 className="size-4 animate-spin" /> : <Activity className="size-4" />}
            Simulate RF Environment
          </Button>
        )}
        <Button type="submit" disabled={pending} className="gap-2">
          {pending ? <Loader2 className="size-4 animate-spin" /> : <Radar className="size-4" />}
          {pending ? "Running model…" : "Predict availability"}
        </Button>
        <p className="text-xs text-muted-foreground">
          Sends POST /api/predict to the existing Flask + Random Forest service.
        </p>
      </div>
    </form>
  );
}
