import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { SimulationChannel } from "@/types/wire-watcher";

const STATE_COLORS: Record<string, string> = {
  OCCUPIED: "#ef4444",
  AVAILABLE: "#22c55e",
  UNAVAILABLE: "#f59e0b",
  ALLOCATED: "#3b82f6",
};

export function ChannelSpectrumChart({
  frequencies,
  powers,
  channels,
  noiseFloor,
}: {
  frequencies: number[];
  powers: number[];
  channels: SimulationChannel[];
  noiseFloor: number;
}) {
  const data = frequencies.map((freq, i) => ({ frequency: freq, power: powers[i] }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="rounded border border-border bg-surface p-2 text-xs shadow-md">
          <p className="font-semibold">{`${Number(label).toFixed(2)} MHz`}</p>
          <p className="text-muted-foreground">{`Power: ${payload[0].value.toFixed(1)} dBm`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="panel px-4 py-4 lg:px-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">Composite Spectrum — Power Spectral Density</h3>
        <div className="flex flex-wrap gap-3 text-xs">
          {Object.entries(STATE_COLORS).map(([state, color]) => (
            <span key={state} className="flex items-center gap-1.5">
              <span className="size-2 rounded-full" style={{ backgroundColor: color }} />
              {state}
            </span>
          ))}
        </div>
      </div>
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" vertical={false} />
            <XAxis
              dataKey="frequency"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v) => v.toFixed(0)}
              fontSize={11}
              className="fill-muted-foreground"
            />
            <YAxis
              domain={["dataMin - 10", "dataMax + 10"]}
              tickFormatter={(v) => `${v}`}
              fontSize={11}
              className="fill-muted-foreground"
            />
            <Tooltip content={<CustomTooltip />} />

            {channels.map((c) => (
              <ReferenceArea
                key={c.channel_id}
                x1={c.start_mhz}
                x2={c.end_mhz}
                fill={STATE_COLORS[c.state] ?? "#94a3b8"}
                fillOpacity={0.1}
                stroke={STATE_COLORS[c.state] ?? "#94a3b8"}
                strokeOpacity={0.3}
              />
            ))}

            <ReferenceLine
              y={noiseFloor}
              stroke="#ef4444"
              strokeDasharray="3 3"
              label={{ value: `Noise Floor: ${noiseFloor.toFixed(1)} dBm`, position: "insideTopLeft", fontSize: 10, fill: "#ef4444" }}
            />

            <Line type="monotone" dataKey="power" stroke="#3b82f6" strokeWidth={1.25} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
