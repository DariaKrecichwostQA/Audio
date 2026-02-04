
import React from 'react';
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, AreaChart, Area, Brush, ReferenceArea 
} from 'recharts';
import { AudioChartData, Anomaly } from '../types';

interface Props {
  data: AudioChartData[];
  threshold: number;
  anomalies: Anomaly[];
  currentTime?: number;
  onPointClick?: (point: AudioChartData) => void;
}

// Added threshold to the destructured props to allow access within the tooltip component
const CustomTooltip = ({ active, payload, threshold }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900 border border-slate-700 p-3 rounded-xl shadow-2xl backdrop-blur-lg border-l-4 border-l-indigo-500 z-[300]">
        <p className="text-[10px] text-slate-400 font-black uppercase mb-2">Punkt Analizy</p>
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between items-center gap-4">
            <span className="text-[9px] text-slate-500 uppercase font-bold">Anomalia</span>
            <span className={`text-xs font-mono font-black ${payload[1].value > threshold ? 'text-red-400' : 'text-indigo-400'}`}>
              {payload[1].value.toFixed(3)}
            </span>
          </div>
          <div className="flex justify-between items-center gap-4">
            <span className="text-[9px] text-slate-500 uppercase font-bold">Hz (Centroid)</span>
            <span className="text-xs font-mono font-black text-emerald-400">
              {payload[0].value.toFixed(0)} Hz
            </span>
          </div>
          <div className="text-[9px] text-slate-600 font-mono mt-1 italic">
            Sekunda: {payload[0].payload.time}s
          </div>
        </div>
      </div>
    );
  }
  return null;
};

const AnomalyChart: React.FC<Props> = ({ data, threshold, anomalies, currentTime, onPointClick }) => {
  // Obliczamy dynamiczne maksimum dla osi Y, aby alarm był zawsze widoczny
  const maxScore = Math.max(...data.map(d => d.anomalyLevel), threshold);
  const yDomain = [0, maxScore * 1.2];

  return (
    <div className="w-full h-full min-h-[400px] flex flex-col">
      <div className="flex-1 relative pb-16">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart 
            data={data} 
            onClick={(e) => e && e.activePayload && onPointClick?.(e.activePayload[0].payload)}
            margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
          >
            <defs>
              <linearGradient id="colorAnom" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorHz" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
              </linearGradient>
            </defs>
            
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            
            <XAxis 
              dataKey="time" 
              hide={data.length > 100} 
              stroke="#475569"
              fontSize={10}
            />
            
            <YAxis 
              yAxisId="left"
              domain={[0, 20000]}
              hide
            />

            <YAxis 
              yAxisId="right"
              orientation="right"
              domain={yDomain}
              stroke="#ef4444"
              fontSize={10}
              tickFormatter={(v) => v.toFixed(1)}
            />
            
            {/* Added threshold prop here so it's passed to the custom tooltip component */}
            <Tooltip content={<CustomTooltip threshold={threshold} />} />
            
            {anomalies.map((anom) => {
              const startIdx = data.findIndex(d => Math.abs((d.second || 0) - (anom.offsetSeconds || 0)) < 0.1);
              const endIdx = data.findIndex(d => Math.abs((d.second || 0) - ((anom.offsetSeconds || 0) + anom.durationSeconds)) < 0.1);
              if (startIdx !== -1 && endIdx !== -1) {
                return (
                  <ReferenceArea 
                    key={anom.id}
                    x1={data[startIdx].time}
                    x2={data[endIdx].time}
                    fill="#ef4444"
                    fillOpacity={0.15}
                    stroke="none"
                  />
                );
              }
              return null;
            })}

            <Area 
              yAxisId="left"
              type="monotone" 
              dataKey="amplitude" 
              stroke="#10b981" 
              fillOpacity={1} 
              fill="url(#colorHz)" 
              strokeWidth={1}
              isAnimationActive={false}
              name="Hz"
            />
            
            <Area 
              yAxisId="right"
              type="monotone" 
              dataKey="anomalyLevel" 
              stroke="#ef4444" 
              fillOpacity={1} 
              fill="url(#colorAnom)" 
              strokeWidth={3}
              isAnimationActive={false}
              name="Anomalia"
            />

            <ReferenceLine 
              yAxisId="right"
              y={threshold} 
              stroke="#ef4444" 
              strokeWidth={2}
              strokeDasharray="5 5" 
              label={{ value: 'ALARM', position: 'insideRight', fill: '#ef4444', fontSize: 10, fontWeight: 'black', offset: 10 }} 
            />
            
            {currentTime !== undefined && (
              <ReferenceLine 
                x={data.find(d => Math.abs((d.second || 0) - currentTime) < 0.05)?.time} 
                stroke="#6366f1" 
                strokeWidth={2}
              />
            )}

            <Brush 
              dataKey="time" 
              height={40} 
              stroke="#4f46e5" 
              fill="#0f172a" 
              gap={1}
              travellerWidth={20}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-8 flex flex-wrap justify-between items-center px-4 gap-4 bg-slate-900/40 p-3 rounded-2xl border border-slate-800/50">
        <div className="flex gap-4">
           <div className="flex items-center gap-2">
             <div className="w-2 h-2 bg-emerald-500 rounded-full shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
             <span className="text-[9px] font-black uppercase text-slate-400">Dźwięk (Hz)</span>
           </div>
           <div className="flex items-center gap-2">
             <div className="w-2 h-2 bg-red-500 rounded-full shadow-[0_0_8px_rgba(239,68,68,0.5)]"></div>
             <span className="text-[9px] font-black uppercase text-slate-400">Anomalia</span>
           </div>
        </div>
        <p className="text-[8px] font-black uppercase text-slate-500 tracking-[0.2em] animate-pulse">
          Przesuń niebieski suwak, aby przybliżyć (ZOOM)
        </p>
      </div>
    </div>
  );
};

export default AnomalyChart;
