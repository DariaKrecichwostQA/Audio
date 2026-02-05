
import React from 'react';
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, AreaChart, Area, Brush, ReferenceArea 
} from 'recharts';
// Add Zap import from lucide-react
import { Zap } from 'lucide-react';
import { AudioChartData, Anomaly } from '../types';

interface Props {
  data: AudioChartData[];
  threshold: number;
  anomalies: Anomaly[];
  currentTime?: number;
  onPointClick?: (point: AudioChartData) => void;
}

const CustomTooltip = ({ active, payload, threshold }: any) => {
  if (active && payload && payload.length) {
    const isAnomaly = payload[1].value > threshold;
    return (
      <div className={`bg-slate-900 border p-3 rounded-xl shadow-2xl backdrop-blur-lg ${isAnomaly ? 'border-red-500 border-l-4' : 'border-slate-700 border-l-4 border-l-indigo-500'} z-[300]`}>
        <p className="text-[10px] text-slate-400 font-black uppercase mb-2">Diagnostyka Czasu</p>
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between items-center gap-4">
            <span className="text-[9px] text-slate-500 uppercase font-bold">Współczynnik błędu</span>
            <span className={`text-xs font-mono font-black ${isAnomaly ? 'text-red-400' : 'text-indigo-400'}`}>
              {payload[1].value.toFixed(3)}
            </span>
          </div>
          <div className="flex justify-between items-center gap-4">
            <span className="text-[9px] text-slate-500 uppercase font-bold">Centroid widma</span>
            <span className="text-xs font-mono font-black text-emerald-400">
              {payload[0].value.toFixed(0)} Hz
            </span>
          </div>
          <div className="text-[9px] text-slate-600 font-mono mt-1 italic">
            Pozycja: {payload[0].payload.time}s
          </div>
        </div>
      </div>
    );
  }
  return null;
};

const AnomalyChart: React.FC<Props> = ({ data, threshold, anomalies, currentTime, onPointClick }) => {
  const maxScore = Math.max(...data.map(d => d.anomalyLevel), threshold, 5);
  const yDomain = [0, maxScore * 1.1];

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
              hide={data.length > 80} 
              stroke="#475569"
              fontSize={9}
              tick={{fill: '#475569'}}
            />
            
            <YAxis yAxisId="left" hide />
            <YAxis 
              yAxisId="right"
              orientation="right"
              domain={yDomain}
              stroke="#ef4444"
              fontSize={9}
              tick={{fill: '#ef4444'}}
              tickFormatter={(v) => v.toFixed(1)}
            />
            
            <Tooltip content={<CustomTooltip threshold={threshold} />} />
            
            {/* Rysowanie segmentów anomalii jako obszarów referencyjnych */}
            {anomalies.map((anom) => {
              // Szukamy najbliższych etykiet czasu na osi X dla segmentu
              const startLabel = data.reduce((prev, curr) => 
                Math.abs((curr.second || 0) - (anom.offsetSeconds || 0)) < Math.abs((prev.second || 0) - (anom.offsetSeconds || 0)) ? curr : prev
              , data[0])?.time;

              const endTime = (anom.offsetSeconds || 0) + anom.durationSeconds;
              const endLabel = data.reduce((prev, curr) => 
                Math.abs((curr.second || 0) - endTime) < Math.abs((prev.second || 0) - endTime) ? curr : prev
              , data[0])?.time;

              if (startLabel && endLabel) {
                return (
                  <ReferenceArea 
                    key={anom.id}
                    x1={startLabel}
                    x2={endLabel}
                    fill="#ef4444"
                    fillOpacity={0.2}
                    stroke="#ef4444"
                    strokeOpacity={0.4}
                    strokeDasharray="3 3"
                    className="cursor-pointer"
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
              strokeWidth={2}
              isAnimationActive={false}
              name="Anomalia"
            />

            <ReferenceLine 
              yAxisId="right"
              y={threshold} 
              stroke="#ef4444" 
              strokeWidth={1.5}
              strokeDasharray="10 5" 
              label={{ value: 'PRÓG NIEREGULARNOŚCI', position: 'insideRight', fill: '#ef4444', fontSize: 9, fontWeight: 'black', offset: 10 }} 
            />
            
            {currentTime !== undefined && (
              <ReferenceLine 
                x={data.reduce((prev, curr) => Math.abs((curr.second || 0) - currentTime) < Math.abs((prev.second || 0) - currentTime) ? curr : prev, data[0])?.time} 
                stroke="#6366f1" 
                strokeWidth={2}
                isAnimationActive={false}
              />
            )}

            <Brush 
              dataKey="time" 
              height={30} 
              stroke="#4f46e5" 
              fill="#0f172a" 
              gap={1}
              travellerWidth={15}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-8 flex flex-wrap justify-between items-center px-4 gap-4 bg-slate-900/40 p-3 rounded-2xl border border-slate-800/50">
        <div className="flex gap-4">
           <div className="flex items-center gap-2">
             <div className="w-2 h-2 bg-emerald-500 rounded-full shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
             <span className="text-[9px] font-black uppercase text-slate-400 tracking-wider">Charakterystyka</span>
           </div>
           <div className="flex items-center gap-2">
             <div className="w-2 h-2 bg-red-500 rounded-full shadow-[0_0_8px_rgba(239,68,68,0.5)]"></div>
             <span className="text-[9px] font-black uppercase text-slate-400 tracking-wider">Anomalia (Segment)</span>
           </div>
        </div>
        <div className="flex items-center gap-2 bg-slate-950/50 px-3 py-1 rounded-full border border-slate-800">
           <Zap className="w-3 h-3 text-indigo-400" />
           <p className="text-[8px] font-black uppercase text-slate-500 tracking-[0.15em]">
              Interaktywny Segment: Kliknij czerwone pole, aby przejść do zdarzenia
           </p>
        </div>
      </div>
    </div>
  );
};

export default AnomalyChart;
