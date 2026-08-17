import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceDot
} from 'recharts';

interface SpectrumChartProps {
  frequencies: number[];
  powers: number[];
  noiseFloor: number;
  peaks: { frequency_mhz: number; power_dbm: number; bandwidth_mhz: number }[];
  centerFreq: number;
  bandwidth: number;
}

export const SpectrumChart: React.FC<SpectrumChartProps> = ({
  frequencies,
  powers,
  noiseFloor,
  peaks,
  centerFreq,
  bandwidth
}) => {
  // Format data for Recharts
  const data = frequencies.map((freq, i) => ({
    frequency: freq,
    power: powers[i]
  }));

  const startFreq = centerFreq - bandwidth / 2;
  const endFreq = centerFreq + bandwidth / 2;

  // Custom Tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 p-2 border border-slate-700 rounded shadow-md text-xs text-slate-200">
          <p className="font-bold">{`${label.toFixed(3)} MHz`}</p>
          <p>{`Power: ${payload[0].value.toFixed(1)} dBm`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-64 bg-slate-900 rounded-lg p-4 border border-slate-800">
      <h3 className="text-sm font-semibold text-slate-300 mb-2">Power Spectral Density (PSD)</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
          <XAxis 
            dataKey="frequency" 
            type="number"
            domain={['dataMin', 'dataMax']} 
            tickFormatter={(val) => val.toFixed(1)}
            stroke="#94a3b8"
            fontSize={12}
          />
          <YAxis 
            domain={['dataMin - 10', 'dataMax + 10']} 
            stroke="#94a3b8"
            fontSize={12}
            tickFormatter={(val) => `${val} dBm`}
          />
          <Tooltip content={<CustomTooltip />} />
          
          {/* Signal Line */}
          <Line 
            type="monotone" 
            dataKey="power" 
            stroke="#3b82f6" 
            strokeWidth={1.5} 
            dot={false}
            isAnimationActive={false} 
          />
          
          {/* Noise Floor Reference */}
          <ReferenceLine 
            y={noiseFloor} 
            stroke="#ef4444" 
            strokeDasharray="3 3" 
            label={{ position: 'insideTopLeft', value: `Noise Floor: ${noiseFloor.toFixed(1)} dBm`, fill: '#ef4444', fontSize: 11 }}
          />
          
          {/* Detected Peaks */}
          {peaks.map((peak, idx) => (
            <ReferenceDot
              key={`peak-${idx}`}
              x={peak.frequency_mhz}
              y={peak.power_dbm}
              r={4}
              fill="#22c55e"
              stroke="white"
            />
          ))}
          
          {/* Band of Interest Highlight - using two reference lines instead of ReferenceArea to keep it simple */}
          <ReferenceLine x={startFreq} stroke="#22c55e" strokeOpacity={0.5} strokeDasharray="3 3" />
          <ReferenceLine x={endFreq} stroke="#22c55e" strokeOpacity={0.5} strokeDasharray="3 3" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
