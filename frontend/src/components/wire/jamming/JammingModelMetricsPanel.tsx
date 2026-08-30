import { Panel } from "@/components/wire/Panel";
import type { JammingModelInfoResponse } from "@/types/wire-watcher";

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div>
      <p className="label-caps text-[10px]">{label}</p>
      <p className={`numeric mt-0.5 text-sm font-semibold ${highlight ? "text-primary" : "text-foreground"}`}>{value}</p>
    </div>
  );
}

export function JammingModelMetricsPanel({ info }: { info: JammingModelInfoResponse }) {
  const m = info.primary_controlled_metrics;

  return (
    <Panel title="Controlled Validation Performance" subtitle={m.description}>
      <div className="grid grid-cols-3 gap-4 sm:grid-cols-6">
        <Stat label="Accuracy" value={`${(m.accuracy * 100).toFixed(1)}%`} />
        <Stat label="Precision" value={`${(m.precision * 100).toFixed(1)}%`} />
        <Stat label="Recall" value={`${(m.recall * 100).toFixed(1)}%`} />
        <Stat label="F1" value={m.f1.toFixed(3)} highlight />
        <Stat label="ROC-AUC" value={m.roc_auc.toFixed(3)} highlight />
        <Stat label="PR-AUC" value={m.pr_auc.toFixed(3)} />
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        Evaluated on {m.n.toLocaleString()} held-out samples never seen during training, restricted to
        a single collection environment so the environment itself can't explain the score. A separate,
        environment-confounded raw test split reports ROC-AUC = {info.supplementary_raw_test_metrics.roc_auc.toFixed(2)} —
        that number is not used as the headline result because it partly reflects which environment a
        sample came from, not just its RF content.
      </p>
    </Panel>
  );
}
