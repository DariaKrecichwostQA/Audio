
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip } from 'recharts';

interface Props {
  frames: number[][];
}

const SpectralAnalysisChart: React.FC<Props> = ({ frames }) => {
  if (!frames || frames.length === 0) return null;
  
  // Flattening or picking middle frame for visualization
  const targetFrame = frames[Math.floor(frames.length / 2)];
  const data = targetFrame.map((val, i) => ({ freq: `${(i * 172).toFixed(0)}Hz`, val }));

  return (
    <div className="w-full h-48 bg-slate-950 rounded-2xl p-4 border border-slate-800">
      <h4 className="text-[9px] font-black uppercase text-slate-500 mb-2 tracking-widest">Analiza Widmowa Incydentu</h4>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <Tooltip 
            cursor={{ fill: 'rgba(99, 102, 241, 0.1)' }}
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="bg-slate-900 border border-slate-700 p-2 rounded shadow text-[10px] font-mono text-indigo-400">
                    {payload[0].value?.toLocaleString()} @ {payload[0].payload.freq}
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar dataKey="val">
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={index > 30 ? '#ef4444' : '#6366f1'} />
            ))}
          </Bar>
          <XAxis dataKey="freq" hide />
          <YAxis hide />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SpectralAnalysisChart;
