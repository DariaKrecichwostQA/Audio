
import React, { useState } from 'react';
import { Anomaly } from '../types';
import { FileText, Download, ShieldAlert, Zap, Clock, Activity, MessageSquare } from 'lucide-react';
import { geminiService } from '../services/geminiService';

interface Props {
  anomalies: Anomaly[];
}

const ReportTable: React.FC<Props> = ({ anomalies }) => {
  const [aiAnalysis, setAiAnalysis] = useState<string | null>(null);
  const [loadingAi, setLoadingAi] = useState(false);

  const getAiInsights = async () => {
    setLoadingAi(true);
    const analysis = await geminiService.analyzeAnomalies(anomalies);
    setAiAnalysis(analysis);
    setLoadingAi(false);
  };

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-800/50 p-6 rounded-3xl border border-slate-700">
           <p className="text-[10px] font-black uppercase text-slate-500 mb-2">Całkowity czas anomalii</p>
           <p className="text-2xl font-black text-white">
              {anomalies.reduce((acc, curr) => acc + curr.durationSeconds, 0).toFixed(2)}s
           </p>
        </div>
        <div className="bg-slate-800/50 p-6 rounded-3xl border border-slate-700">
           <p className="text-[10px] font-black uppercase text-slate-500 mb-2">Średnia intensywność</p>
           <p className="text-2xl font-black text-indigo-400">
              {(anomalies.reduce((acc, curr) => acc + curr.intensity, 0) / (anomalies.length || 1)).toFixed(2)}x
           </p>
        </div>
        <div className="bg-slate-800/50 p-6 rounded-3xl border border-slate-700">
           <p className="text-[10px] font-black uppercase text-slate-500 mb-2">Status Maszyny</p>
           <div className="flex items-center gap-3">
              <div className={`w-4 h-4 rounded-full ${anomalies.length > 3 ? 'bg-red-500 animate-pulse' : 'bg-emerald-500'}`}></div>
              <p className={`text-xl font-black uppercase ${anomalies.length > 3 ? 'text-red-400' : 'text-emerald-400'}`}>
                {anomalies.length > 3 ? 'Krytyczny' : 'Nominalny'}
              </p>
           </div>
        </div>
      </div>

      <div className="bg-slate-950/50 rounded-3xl border border-slate-800 overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-slate-900/80 border-b border-slate-800">
              <th className="p-5 text-[10px] font-black uppercase text-slate-500">ID</th>
              <th className="p-5 text-[10px] font-black uppercase text-slate-500">Czas Startu</th>
              <th className="p-5 text-[10px] font-black uppercase text-slate-500">Czas Trwania</th>
              <th className="p-5 text-[10px] font-black uppercase text-slate-500">Intensywność</th>
              <th className="p-5 text-[10px] font-black uppercase text-slate-500">Stopień</th>
            </tr>
          </thead>
          <tbody>
            {anomalies.map((a, idx) => (
              <tr key={a.id} className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-all">
                <td className="p-5 text-xs font-mono text-indigo-400">#{(idx+1).toString().padStart(3, '0')}</td>
                <td className="p-5 text-xs text-white">{a.offsetSeconds?.toFixed(2)}s</td>
                <td className="p-5 text-xs text-slate-400">{a.durationSeconds.toFixed(3)}s</td>
                <td className="p-5 text-xs font-bold text-red-400">{a.intensity.toFixed(2)}x</td>
                <td className="p-5">
                   <span className={`px-2 py-1 rounded text-[9px] font-black uppercase ${a.severity === 'High' ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'}`}>
                      {a.severity}
                   </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="bg-indigo-600/10 border border-indigo-500/20 p-8 rounded-[2.5rem] relative overflow-hidden">
        <div className="absolute top-0 right-0 p-10 opacity-5 pointer-events-none">
          <MessageSquare size={120} />
        </div>
        <h3 className="text-xl font-black uppercase tracking-tight text-white flex items-center gap-3 mb-6">
          <ShieldAlert className="text-indigo-500" /> Analiza Diagnostyczna Gemini AI
        </h3>
        
        {aiAnalysis ? (
          <div className="prose prose-invert max-w-none">
            <div className="text-sm text-indigo-200 leading-relaxed whitespace-pre-wrap bg-slate-900/50 p-6 rounded-2xl border border-indigo-500/30">
              {aiAnalysis}
            </div>
            <button onClick={() => setAiAnalysis(null)} className="mt-4 text-[10px] font-black uppercase text-slate-500 hover:text-white">Wyczyść analizę</button>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-indigo-500/20 rounded-3xl">
             <p className="text-slate-400 text-sm mb-6 font-medium">Użyj sztucznej inteligencji, aby zinterpretować wzorce dźwiękowe.</p>
             <button 
                onClick={getAiInsights}
                disabled={loadingAi || anomalies.length === 0}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 text-white px-8 py-3 rounded-2xl text-[11px] font-black uppercase shadow-xl shadow-indigo-600/20 flex items-center gap-2 transition-all"
             >
                {loadingAi ? <Activity className="animate-spin w-4 h-4" /> : <Zap className="w-4 h-4 fill-current" />}
                Generuj Diagnozę AI
             </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportTable;
