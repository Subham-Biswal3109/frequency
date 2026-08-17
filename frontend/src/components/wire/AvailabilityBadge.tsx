import { cn } from "@/lib/utils";

export function AvailabilityBadge({
  available,
  className,
}: {
  available: boolean | null;
  className?: string;
}) {
  if (available === null) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-2 rounded-full border border-border bg-muted/40 px-2.5 py-1 text-xs text-muted-foreground",
          className,
        )}
      >
        <span className="size-1.5 rounded-full bg-muted-foreground" />
        NOT REPORTED
      </span>
    );
  }

  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-2.5 py-1 font-mono text-xs tracking-wider",
        available
          ? "border-available/40 bg-available/12 text-available"
          : "border-occupied/40 bg-occupied/12 text-occupied",
        className,
      )}
    >
      <span
        className={cn("size-1.5 rounded-full", available ? "bg-available" : "bg-occupied")}
      />
      {available ? "AVAILABLE" : "OCCUPIED"}
    </span>
  );
}
