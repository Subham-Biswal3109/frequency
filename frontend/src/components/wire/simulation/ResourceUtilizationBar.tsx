import type { ResourceUtilization } from "@/types/wire-watcher";

export function ResourceUtilizationBar({
  before,
  after,
}: {
  before: ResourceUtilization;
  after: ResourceUtilization;
}) {
  const Bar = ({ util, label }: { util: ResourceUtilization; label: string }) => {
    const total = util.total_mhz || 1;
    const occupiedPct = (util.occupied_mhz / total) * 100;
    const allocatedPct = (util.allocated_mhz / total) * 100;
    const availablePct = Math.max(0, 100 - occupiedPct - allocatedPct);

    return (
      <div>
        <div className="mb-1.5 flex items-center justify-between text-xs">
          <span className="label-caps">{label}</span>
          <span className="numeric text-muted-foreground">{util.total_mhz} MHz total</span>
        </div>
        <div className="flex h-6 overflow-hidden rounded-md border border-border">
          <div className="bg-occupied/75" style={{ width: `${occupiedPct}%` }} title={`Occupied: ${util.occupied_mhz} MHz`} />
          <div className="bg-primary/75" style={{ width: `${allocatedPct}%` }} title={`Allocated: ${util.allocated_mhz} MHz`} />
          <div className="bg-available/60" style={{ width: `${availablePct}%` }} title={`Available: ${util.available_mhz} MHz`} />
        </div>
        <div className="numeric mt-1.5 flex gap-4 text-xs text-muted-foreground">
          <span>Occupied: {util.occupied_mhz} MHz</span>
          <span>Allocated: {util.allocated_mhz} MHz</span>
          <span>Available: {util.available_mhz} MHz</span>
        </div>
      </div>
    );
  };

  return (
    <div className="panel space-y-4 px-4 py-4 lg:px-5">
      <h3 className="text-sm font-semibold">Resource Utilization</h3>
      <Bar util={before} label="Before Allocation" />
      <Bar util={after} label="After Allocation" />
    </div>
  );
}
