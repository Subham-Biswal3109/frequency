import { Link } from "@tanstack/react-router";
import {
  Activity,
  BrainCircuit,
  Gauge,
  History,
  Menu,
  Radar,
  ServerCog,
  Waves,
  X,
} from "lucide-react";
import { useState, type ReactNode } from "react";

import { API_BASE_URL } from "@/services/api";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Dashboard", icon: Gauge },
  { to: "/predict", label: "Prediction", icon: Radar },
  { to: "/history", label: "Prediction History", icon: History },
  { to: "/simulation", label: "Spectrum Simulation", icon: Waves },
  { to: "/monitoring", label: "Spectrum Monitoring", icon: Activity },
  { to: "/model", label: "Model Information", icon: BrainCircuit },
  { to: "/status", label: "System Status", icon: ServerCog },
] as const;

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex flex-col gap-1">
      {NAV.map(({ to, label, icon: Icon }) => (
        <Link
          key={to}
          to={to}
          onClick={onNavigate}
          activeOptions={{ exact: to === "/" }}
          className="group flex items-center gap-3 rounded-md px-3 py-2 text-sm text-sidebar-foreground/75 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
          activeProps={{
            className:
              "bg-sidebar-accent text-sidebar-accent-foreground shadow-[inset_2px_0_0_0_var(--color-sidebar-primary)]",
          }}
        >
          <Icon className="size-4 shrink-0 opacity-80" />
          <span className="truncate">{label}</span>
        </Link>
      ))}
    </nav>
  );
}

function Brand() {
  return (
    <Link to="/" className="flex items-center gap-3">
      <span className="relative grid size-9 place-items-center rounded-md border border-primary/40 bg-primary/10">
        <Radar className="size-5 text-primary" />
        <span className="absolute inset-0 animate-pulse rounded-md ring-1 ring-primary/20" />
      </span>
      <span className="leading-tight">
        <span className="block font-display text-sm font-semibold tracking-tight">Wire Watcher</span>
        <span className="label-caps block">Spectrum Intelligence</span>
      </span>
    </Link>
  );
}

export function AppShell({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[260px_1fr]">
      <aside className="sticky top-0 z-30 hidden h-screen flex-col border-r border-sidebar-border bg-sidebar/80 p-4 backdrop-blur lg:flex">
        <Brand />
        <div className="mt-8 flex-1">
          <p className="label-caps mb-3 px-3">Sections</p>
          <NavLinks />
        </div>
        <div className="rounded-md border border-border/70 bg-surface/60 p-3">
          <p className="label-caps">Flask API</p>
          <p className="numeric mt-1 truncate text-xs text-foreground/80">{API_BASE_URL}</p>
        </div>
      </aside>

      <div className="flex min-h-screen flex-col">
        <header className="sticky top-0 z-20 border-b border-border bg-background/85 backdrop-blur">
          <div className="flex items-center gap-3 px-4 py-3 lg:hidden">
            <button
              type="button"
              aria-label={open ? "Close navigation" : "Open navigation"}
              onClick={() => setOpen((v) => !v)}
              className="grid size-9 place-items-center rounded-md border border-border bg-surface"
            >
              {open ? <X className="size-4" /> : <Menu className="size-4" />}
            </button>
            <Brand />
          </div>
          {open ? (
            <div className="border-t border-border px-4 py-3 lg:hidden">
              <NavLinks onNavigate={() => setOpen(false)} />
            </div>
          ) : null}

          <div className="hidden items-end justify-between gap-6 px-6 py-5 lg:flex">
            <div>
              <h1 className="text-xl font-semibold">{title}</h1>
              <p className="mt-1 text-sm text-muted-foreground">{description}</p>
            </div>
            <p className="label-caps max-w-xs text-right leading-relaxed">
              ML-based spectrum availability prediction using synthetic training data
            </p>
          </div>
        </header>

        <main className={cn("flex-1 px-4 py-6 lg:px-6 lg:py-8")}>
          <div className="lg:hidden">
            <h1 className="text-lg font-semibold">{title}</h1>
            <p className="mt-1 mb-5 text-sm text-muted-foreground">{description}</p>
          </div>
          {children}
        </main>

        <footer className="border-t border-border px-4 py-4 text-xs text-muted-foreground lg:px-6">
          Wire Watcher — ML-based spectrum availability prediction using synthetic training data.
          Predictions are model estimates, not real-time spectrum measurements.
        </footer>
      </div>
    </div>
  );
}
