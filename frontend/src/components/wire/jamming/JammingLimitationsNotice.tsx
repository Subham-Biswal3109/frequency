import { AlertTriangle } from "lucide-react";

export function JammingLimitationsNotice({ limitations }: { limitations: string[] }) {
  return (
    <details className="panel px-4 py-3 lg:px-5">
      <summary className="flex cursor-pointer items-center gap-2 text-sm font-medium text-warning">
        <AlertTriangle className="size-4 shrink-0" />
        Dataset &amp; model limitations (click to expand)
      </summary>
      <ul className="mt-3 list-disc space-y-1.5 pl-5 text-xs text-muted-foreground">
        {limitations.map((l, i) => (
          <li key={i}>{l}</li>
        ))}
      </ul>
    </details>
  );
}
