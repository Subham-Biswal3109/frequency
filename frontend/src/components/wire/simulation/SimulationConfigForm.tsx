import { zodResolver } from "@hookform/resolvers/zod";
import { Loader2, Play, Plus, Trash2 } from "lucide-react";
import { useFieldArray, useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Panel } from "@/components/wire/Panel";
import type { SimulationMode, SimulationRunRequest } from "@/types/wire-watcher";

const userSchema = z.object({
  user_id: z.string().trim().min(1, "Required"),
  requested_bandwidth_mhz: z.coerce.number().finite().positive(),
});

const schema = z
  .object({
    start_frequency_mhz: z.coerce.number().finite().positive(),
    end_frequency_mhz: z.coerce.number().finite().positive(),
    channel_bandwidth_mhz: z.coerce.number().finite().positive(),
    noise_floor_dbm: z.coerce.number().finite(),
    num_existing_users: z.coerce.number().int().min(0).max(200),
    seed: z.union([z.coerce.number().int(), z.literal("")]).optional(),
    mode: z.enum(["basic", "ml_assisted", "multi_user"]),
    requested_bandwidth_mhz: z.union([z.coerce.number().finite().positive(), z.literal("")]).optional(),
    users: z.array(userSchema).optional(),
    state: z.string().trim().min(1),
    city: z.string().trim().min(1),
    service_type: z.string().trim().min(1),
  })
  .refine((v) => v.end_frequency_mhz > v.start_frequency_mhz, {
    message: "End frequency must be greater than start frequency",
    path: ["end_frequency_mhz"],
  })
  .refine((v) => v.channel_bandwidth_mhz <= v.end_frequency_mhz - v.start_frequency_mhz, {
    message: "Channel bandwidth cannot exceed the total frequency range",
    path: ["channel_bandwidth_mhz"],
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

export function SimulationConfigForm({
  onSubmit,
  pending,
}: {
  onSubmit: (input: SimulationRunRequest) => void;
  pending: boolean;
}) {
  const {
    register,
    handleSubmit,
    control,
    watch,
    setValue,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      start_frequency_mhz: 1800,
      end_frequency_mhz: 1900,
      channel_bandwidth_mhz: 10,
      noise_floor_dbm: -100,
      num_existing_users: 5,
      seed: 42,
      mode: "ml_assisted",
      requested_bandwidth_mhz: 10,
      users: [
        { user_id: "User A", requested_bandwidth_mhz: 10 },
        { user_id: "User B", requested_bandwidth_mhz: 10 },
      ],
      state: "Maharashtra",
      city: "Mumbai",
      service_type: "4G LTE",
    },
  });

  const { fields, append, remove } = useFieldArray({ control, name: "users" });
  const mode = watch("mode") as SimulationMode;

  const submit = handleSubmit((values) => {
    const parsed = schema.parse(values);
    const payload: SimulationRunRequest = {
      start_frequency_mhz: parsed.start_frequency_mhz,
      end_frequency_mhz: parsed.end_frequency_mhz,
      channel_bandwidth_mhz: parsed.channel_bandwidth_mhz,
      noise_floor_dbm: parsed.noise_floor_dbm,
      num_existing_users: parsed.num_existing_users,
      mode: parsed.mode,
      state: parsed.state,
      city: parsed.city,
      service_type: parsed.service_type,
    };
    if (parsed.seed !== "" && parsed.seed !== undefined) payload.seed = parsed.seed;

    if (parsed.mode === "multi_user") {
      payload.users = (parsed.users ?? []).map((u) => ({
        user_id: u.user_id,
        requested_bandwidth_mhz: u.requested_bandwidth_mhz,
      }));
    } else if (parsed.requested_bandwidth_mhz !== "" && parsed.requested_bandwidth_mhz !== undefined) {
      payload.requested_bandwidth_mhz = parsed.requested_bandwidth_mhz;
    }

    onSubmit(payload);
  });

  return (
    <Panel
      title="Spectrum Simulation Configuration"
      subtitle="Simulated spectrum sensing and availability-based channel allocation — an engineering demonstration, not a live measurement."
    >
      <form onSubmit={submit} className="space-y-5">
        <div className="grid grid-cols-2 gap-4">
          <Field label="Start Frequency (MHz)" error={errors.start_frequency_mhz?.message}>
            <Input type="number" step="any" {...register("start_frequency_mhz")} />
          </Field>
          <Field label="End Frequency (MHz)" error={errors.end_frequency_mhz?.message}>
            <Input type="number" step="any" {...register("end_frequency_mhz")} />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Field label="Channel Bandwidth" error={errors.channel_bandwidth_mhz?.message}>
            <select
              {...register("channel_bandwidth_mhz")}
              className="flex h-9 w-full items-center rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm"
            >
              <option value={5}>5 MHz</option>
              <option value={10}>10 MHz</option>
              <option value={20}>20 MHz</option>
            </select>
          </Field>
          <Field label="Noise Floor (dBm)" error={errors.noise_floor_dbm?.message}>
            <Input type="number" step="any" {...register("noise_floor_dbm")} />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Field label="Existing Users / Signals" hint="1–10 typical" error={errors.num_existing_users?.message}>
            <Input type="number" step="1" min="0" {...register("num_existing_users")} />
          </Field>
          <Field label="Random Seed" hint="Same seed reproduces the same run" error={errors.seed?.message as string | undefined}>
            <Input type="number" step="1" {...register("seed")} />
          </Field>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <Field label="State">
            <Input {...register("state")} />
          </Field>
          <Field label="City">
            <Input {...register("city")} />
          </Field>
          <Field label="Service Type">
            <Input {...register("service_type")} />
          </Field>
        </div>

        <div className="space-y-2">
          <Label className="label-caps">Simulation Mode</Label>
          <Tabs
            value={mode}
            onValueChange={(v) => setValue("mode", v as SimulationMode)}
          >
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="basic">Basic</TabsTrigger>
              <TabsTrigger value="ml_assisted">ML Assisted</TabsTrigger>
              <TabsTrigger value="multi_user">Multi-User</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        {mode !== "multi_user" ? (
          <Field
            label="New User Bandwidth Requirement (MHz)"
            error={errors.requested_bandwidth_mhz?.message as string | undefined}
          >
            <Input type="number" step="any" {...register("requested_bandwidth_mhz")} />
          </Field>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="label-caps">Users (allocated sequentially, in order)</Label>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => append({ user_id: `User ${String.fromCharCode(65 + fields.length)}`, requested_bandwidth_mhz: 10 })}
              >
                <Plus className="mr-1 size-3.5" /> Add user
              </Button>
            </div>
            <div className="space-y-2">
              {fields.map((field, index) => (
                <div key={field.id} className="flex items-end gap-2">
                  <div className="flex-1">
                    <Input placeholder="User ID" {...register(`users.${index}.user_id` as const)} />
                  </div>
                  <div className="w-32">
                    <Input
                      type="number"
                      step="any"
                      placeholder="MHz"
                      {...register(`users.${index}.requested_bandwidth_mhz` as const)}
                    />
                  </div>
                  <Button type="button" variant="ghost" size="icon" onClick={() => remove(index)}>
                    <Trash2 className="size-4 text-muted-foreground" />
                  </Button>
                </div>
              ))}
            </div>
          </div>
        )}

        <Button type="submit" disabled={pending} className="w-full">
          {pending ? <Loader2 className="mr-2 size-4 animate-spin" /> : <Play className="mr-2 size-4" />}
          {pending ? "Running Simulation…" : "Generate Spectrum & Allocate"}
        </Button>
      </form>
    </Panel>
  );
}
