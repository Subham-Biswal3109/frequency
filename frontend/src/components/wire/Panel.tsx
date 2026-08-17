import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export function Panel({
  title,
  subtitle,
  action,
  className,
  bodyClassName,
  children,
}: {
  title?: string;
  subtitle?: string;
  action?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
}) {
  return (
    <section className={cn("panel", className)}>
      {title ? (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-border px-4 py-3 lg:px-5">
          <div>
            <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
            {subtitle ? <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p> : null}
          </div>
          {action}
        </header>
      ) : null}
      <div className={cn("px-4 py-4 lg:px-5", bodyClassName)}>{children}</div>
    </section>
  );
}

export function KpiCard({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "default" | "available" | "occupied" | "primary";
}) {
  const tones: Record<string, string> = {
    default: "text-foreground",
    primary: "text-primary",
    available: "text-available",
    occupied: "text-occupied",
  };
  return (
    <div className="panel relative overflow-hidden px-4 py-4">
      <p className="label-caps">{label}</p>
      <p className={cn("numeric mt-2 text-3xl font-semibold", tones[tone])}>{value}</p>
      {hint ? <p className="mt-1 text-xs text-muted-foreground">{hint}</p> : null}
      <span className="pointer-events-none absolute -right-8 -top-8 size-24 rounded-full bg-primary/5 blur-2xl" />
    </div>
  );
}
