
import React from 'react';
import { Anomaly } from '../types';
import { BrainCircuit, Cpu, Zap, Activity, Info, BarChart3 } from 'lucide-react';

interface Props {
  anomalies: Anomaly[];
  totalTrained: number;
  sensitivity: number;
  threshold: number;
}

const TechnicalReportView: React.FC<Props> = ({ anomalies, totalTrained, sensitivity, threshold }) => {
  return (
    <div id="tech-report" className="bg-white text-slate-900 p-12 max-w-[210mm] mx-auto shadow-2xl print:shadow-none print:p-0">
      <header className="border-b-4 border-indigo-600 pb-8 mb-8 flex justify-between items-end">
        <div>
          <h1 className="text-4xl font-black tracking-tighter uppercase italic">Audio<span className="text-indigo-600">Sentinel</span></h1>
          <p className="text-slate-500 font-bold uppercase tracking-widest text-xs mt-2">System Diagnostyki Wibroakustycznej AI</p>
        </div>
        <div className="text-right">
          <p className="font-mono text-xs text-slate-400">RAPORT TECHNICZNY v1.5</p>
          <p className="font-mono text-xs text-slate-400">DATA: {new Date().toLocaleDateString('pl-PL')}</p>
        </div>
      </header>

      <section className="mb-10">
        <h2 className="text-xl font-black uppercase mb-4 flex items-center gap-2 border-b border-slate-200 pb-2">
          <BrainCircuit className="w-5 h-5 text-indigo-600" /> 1. Architektura Modelu AI
        </h2>
        <div className="grid grid-cols-2 gap-8 text-sm leading-relaxed">
          <div>
            <p className="font-bold mb-2">Sieć Rekurencyjna LSTM Autoencoder</p>
            <p>System wykorzystuje głęboką sieć neuronową typu <strong>Long Short-Term Memory (LSTM)</strong> zorganizowaną w strukturze autoenkodera. Model składa się z enkodera kompresującego okno czasowe (12 ramek widmowych) do przestrzeni ukrytej (Latent Space: 24 jednostki) oraz dekodera próbującego odtworzyć sygnał pierwotny.</p>
          </div>
          <ul className="bg-slate-50 p-4 rounded-xl space-y-2 list-disc list-inside font-medium text-slate-700">
            <li>Wejście: Tensor [12, 128] (Czas, Częstotliwość)</li>
            <li>Latent Dimension: 24 (Wektor kontekstu)</li>
            <li>Funkcja straty: Mean Squared Error (MSE)</li>
            <li>Optymalizator: Adam (LR: 0.001)</li>
          </ul>
        </div>
      </section>

      <section className="mb-10">
        <h2 className="text-xl font-black uppercase mb-4 flex items-center gap-2 border-b border-slate-200 pb-2">
          <Zap className="w-5 h-5 text-indigo-600" /> 2. Przetwarzanie Sygnału i Fourier
        </h2>
        <div className="text-sm space-y-4">
          <p>Przed podaniem danych do sieci neuronowej, sygnał audio przechodzi przez szereg transformacji matematycznych w celu uwypuklenia cech diagnostycznych:</p>
          <div className="grid grid-cols-3 gap-4">
            <div className="border border-slate-100 p-3 rounded-lg">
              <p className="font-black text-[10px] text-indigo-600 uppercase mb-1">STFT</p>
              <p className="text-[11px]">Krótkoczasowa Transformata Fouriera (256 binów) rozkładająca sygnał na dziedzinę częstotliwościową.</p>
            </div>
            <div className="border border-slate-100 p-3 rounded-lg">
              <p className="font-black text-[10px] text-indigo-600 uppercase mb-1">Pre-emfaza</p>
              <p className="text-[11px]">Filtr górnoprzepustowy wzmacniający składowe wysokotonowe (tarcie, pisk) kosztem niskotonowego szumu.</p>
            </div>
            <div className="border border-slate-100 p-3 rounded-lg">
              <p className="font-black text-[10px] text-indigo-600 uppercase mb-1">Log-Scaling</p>
              <p className="text-[11px]">Kompresja logarytmiczna dynamiki (skala Mel-podobna) stabilizująca błąd rekonstrukcji.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-10">
        <h2 className="text-xl font-black uppercase mb-4 flex items-center gap-2 border-b border-slate-200 pb-2">
          <Activity className="w-5 h-5 text-indigo-600" /> 3. Detekcja i Próg Statystyczny
        </h2>
        <p className="text-sm mb-4 italic text-slate-600">Algorytm Robust Thresholding oparty na MAD (Median Absolute Deviation):</p>
        <div className="bg-indigo-600 text-white p-6 rounded-2xl font-mono text-xs leading-relaxed">
          Threshold = Baseline + (Sensitivity_Multiplier * MAD * 1.4826)
        </div>
        <div className="mt-4 grid grid-cols-2 gap-4 text-xs font-medium uppercase text-slate-500">
           <div className="flex justify-between border-b pb-1"><span>Aktualna Czułość:</span> <span className="text-slate-900">{sensitivity}</span></div>
           <div className="flex justify-between border-b pb-1"><span>Próg Obliczony:</span> <span className="text-slate-900">{threshold.toFixed(4)}</span></div>
        </div>
      </section>

      <section className="mb-10">
        <h2 className="text-xl font-black uppercase mb-4 flex items-center gap-2 border-b border-slate-200 pb-2">
          <BarChart3 className="w-5 h-5 text-indigo-600" /> 4. Interpretacja Wyników (Wykres)
        </h2>
        <div className="text-sm space-y-4">
          <div className="flex gap-4">
            <div className="w-4 h-4 bg-red-500 rounded-sm shrink-0"></div>
            <div>
              <p className="font-bold">Score (Czerwona Linia)</p>
              <p className="text-[11px]">Reprezentuje błąd rekonstrukcji. Nagłe piki powyżej linii przerywanej oznaczają brak zgodności dźwięku z wyuczonym wzorcem (Anomalia).</p>
            </div>
          </div>
          <div className="flex gap-4">
            <div className="w-4 h-4 bg-emerald-500 rounded-sm shrink-0"></div>
            <div>
              <p className="font-bold">Centroid (Zielona Linia)</p>
              <p className="text-[11px]">Środek ciężkości widma w Hz. Wzrost tej wartości przy jednoczesnym skoku Score wskazuje na usterki mechaniczne o wysokiej częstotliwości (tarcie, brak smarowania).</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-black uppercase mb-4 flex items-center gap-2 border-b border-slate-200 pb-2">
          <Info className="w-5 h-5 text-indigo-600" /> 5. Statystyka Ostatniej Sesji
        </h2>
        <div className="grid grid-cols-3 gap-6 text-center">
          <div className="bg-slate-50 p-4 rounded-xl">
             <p className="text-[10px] font-black uppercase text-slate-400">Wykryte anomalie</p>
             <p className="text-2xl font-black text-slate-900">{anomalies.length}</p>
          </div>
          <div className="bg-slate-50 p-4 rounded-xl">
             <p className="text-[10px] font-black uppercase text-slate-400">Czas anomalii (total)</p>
             <p className="text-2xl font-black text-slate-900">{anomalies.reduce((a,c) => a+c.durationSeconds, 0).toFixed(2)}s</p>
          </div>
          <div className="bg-slate-50 p-4 rounded-xl">
             <p className="text-[10px] font-black uppercase text-slate-400">Pliki treningowe</p>
             <p className="text-2xl font-black text-slate-900">{totalTrained}</p>
          </div>
        </div>
      </section>

      <footer className="mt-16 pt-8 border-t border-slate-100 text-[9px] text-slate-400 flex justify-between items-center uppercase font-bold tracking-widest">
        <span>Projekt: AudioSentinel Diagnostics</span>
        <span>Lokalne Przetwarzanie TensorFlow.js</span>
        <span>Strona 1 / 1</span>
      </footer>
    </div>
  );
};

export default TechnicalReportView;
