import { CheckCircle2 } from "lucide-react";

const STEPS = [
  "Spectrum generated",
  "Occupancy detected (RF sensing)",
  "ML analysis complete",
  "Candidate channels identified",
  "Best channel selected",
];

/**
 * This is a single synchronous REST call (POST /api/simulation/run), not a
 * multi-stage async job — so all steps are shown as already-completed for
 * the run just returned, rather than an artificially delayed live
 * progress bar. Being upfront about this avoids implying a live pipeline
 * that doesn't actually exist.
 */
export function AllocationSteps({ modelLoaded }: { modelLoaded: boolean }) {
  const steps = modelLoaded ? STEPS : STEPS.filter((s) => !s.startsWith("ML"));
  return (
    <div className="panel px-4 py-4 lg:px-5">
      <h3 className="mb-3 text-sm font-semibold">Simulation Pipeline</h3>
      <ul className="space-y-2">
        {steps.map((step) => (
          <li key={step} className="flex items-center gap-2 text-sm text-foreground/85">
            <CheckCircle2 className="size-4 shrink-0 text-available" />
            {step}
          </li>
        ))}
      </ul>
      {!modelLoaded ? (
        <p className="mt-3 text-xs text-warning">
          ML model was not loaded on the backend — RF sensing ran, but the ML availability estimation step was skipped.
        </p>
      ) : null}
    </div>
  );
}
