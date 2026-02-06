
import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, Square, BrainCircuit, Loader2, 
  Settings, Upload, FileAudio, History, Download, 
  Zap, Volume2, FileSearch, Mic, Trash2, 
  ChevronDown, ChevronRight, Terminal, Database,
  SlidersHorizontal, Save, RefreshCw, BarChart3, Cpu, AlertTriangle,
  HardDrive, FolderOpen, Sliders, FileText, Printer, X, MonitorDown,
  Info, FileUp
} from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import { Anomaly, AudioChartData, ModelConfig } from './types';
import AnomalyChart from './components/AnomalyChart';
import ReportTable from './components/ReportTable';
import TechnicalReportView from './components/TechnicalReportView';
import { detector } from './services/anomalyModel';

interface FileQueueItem {
  file: File;
  label: 'Normal' | 'Anomaly';
  status: 'pending' | 'processing' | 'done' | 'error';
}

interface LogEntry {
  message: string;
  type: 'info' | 'success' | 'error' | 'warning';
  time: string;
}

const App: React.FC = () => {
  const [mode, setMode] = useState<'IDLE' | 'TRAINING' | 'MONITORING' | 'FILE_ANALYSIS'>('IDLE');
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [chartData, setChartData] = useState<AudioChartData[]>([]);
  const [status, setStatus] = useState('Inicjalizacja...');
  const [tfBackend, setTfBackend] = useState('...');
  const [showSettings, setShowSettings] = useState(false);
  const [showReportModal, setShowReportModal] = useState(false);
  const [trainingQueue, setTrainingQueue] = useState<FileQueueItem[]>([]);
  const [batchProgress, setBatchProgress] = useState(0);
  const [currentScore, setCurrentScore] = useState(0);
  const [analyzedFileUrl, setAnalyzedFileUrl] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [totalTrained, setTotalTrained] = useState(0);
  const [ramUsage, setRamUsage] = useState(0);
  const [sensitivity, setSensitivity] = useState(detector.getSensitivity());
  const [dynamicThreshold, setDynamicThreshold] = useState(detector.getThreshold());
  const [installPrompt, setInstallPrompt] = useState<any>(null);
  const [isStandalone, setIsStandalone] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [showConsole, setShowConsole] = useState(true);

  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mainAudioRef = useRef<HTMLAudioElement | null>(null);
  const monitoringRef = useRef<boolean>(false);
  const liveBufferRef = useRef<number[][]>([]);
  
  const importJsonRef = useRef<HTMLInputElement>(null);
  const importWeightsRef = useRef<HTMLInputElement>(null);
  const importStatsRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let memInterval: any;
    const init = async () => {
      try {
        await tf.ready();
        await new Promise(r => setTimeout(r, 200));
        setTfBackend(detector.getBackendName());
        setTotalTrained(detector.getTotalFiles());
        setIsStandalone(window.matchMedia('(display-mode: standalone)').matches);
        setStatus('Gotowy');
        memInterval = setInterval(() => {
          const info = detector.getMemInfo();
          if (info && (info as any).numBytes !== undefined) setRamUsage((info as any).numBytes);
        }, 3000);
        addLog("System Desktop AI aktywny", "success");
      } catch (err) { setStatus('Błąd'); }
    };
    init();
    const handlePrompt = (e: any) => { e.preventDefault(); setInstallPrompt(e); };
    window.addEventListener('beforeinstallprompt', handlePrompt);
    return () => { if (memInterval) clearInterval(memInterval); window.removeEventListener('beforeinstallprompt', handlePrompt); };
  }, []);

  const addLog = (message: string, type: LogEntry['type'] = 'info') => {
    setLogs(prev => [...prev.slice(-49), {
      message, type,
      time: new Date().toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' })
    }]);
  };

  const decodeAudioSafe = async (file: File): Promise<AudioBuffer> => {
    const arrayBuffer = await file.arrayBuffer();
    const tempCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    try {
      const decoded = await tempCtx.decodeAudioData(arrayBuffer);
      tempCtx.close();
      return decoded;
    } catch (e) {
      tempCtx.close();
      throw new Error("Błąd formatu WAV");
    }
  };

  const handleImport = async () => {
    const jsonFile = importJsonRef.current?.files?.[0];
    const weightsFile = importWeightsRef.current?.files?.[0];
    const statsFile = importStatsRef.current?.files?.[0];

    if (!jsonFile || !weightsFile) {
      alert("Musisz wybrać co najmniej plik model.json oraz model.weights.bin");
      return;
    }

    setStatus('Importowanie...');
    const success = await detector.importModelFromFiles(jsonFile, weightsFile, statsFile);
    if (success) {
      setTotalTrained(detector.getTotalFiles());
      setDynamicThreshold(detector.getThreshold());
      setSensitivity(detector.getSensitivity());
      addLog("Pomyślnie zaimportowano bazę wiedzy z pliku", "success");
      setShowSettings(false);
    } else {
      addLog("Błąd podczas importu plików modelu", "error");
    }
    setStatus('Gotowy');
  };

  const runIncrementalTraining = async () => {
    if (trainingQueue.length === 0) return;
    setMode('TRAINING');
    addLog(`Uczenie: ${trainingQueue.length} plików`, "warning");
    try {
      for (let i = 0; i < trainingQueue.length; i++) {
        const item = trainingQueue[i];
        setBatchProgress(Math.round((i / trainingQueue.length) * 100));
        try {
          const audioBuffer = await decodeAudioSafe(item.file);
          const rawData = audioBuffer.getChannelData(0);
          const spectrogram = tf.tidy(() => {
            const tensor = tf.tensor1d(rawData);
            const frameStep = Math.max(1, Math.floor(audioBuffer.sampleRate * 0.015));
            return tf.abs(tf.signal.stft(tensor, 256, frameStep));
          });
          const spectrogramData = await spectrogram.array() as number[][];
          spectrogram.dispose();
          const frames = spectrogramData.map(row => detector.scaleFrame(row.slice(0, 128)));
          await detector.trainOnFile(frames, item.label);
          addLog(`Nauczono: ${item.file.name}`, 'success');
        } catch (err: any) { addLog(`Pominięto ${item.file.name}: ${err.message}`, 'error'); }
        await tf.nextFrame();
      }
    } finally {
      setTrainingQueue([]);
      setMode('IDLE');
      setTotalTrained(detector.getTotalFiles());
      setDynamicThreshold(detector.getThreshold());
    }
  };

  const analyzeSingleFile = async (file: File) => {
    try {
      setMode('FILE_ANALYSIS');
      setChartData([]); setAnomalies([]);
      const audioBuffer = await decodeAudioSafe(file);
      const fileUrl = URL.createObjectURL(file);
      setAnalyzedFileUrl(fileUrl);
      const rawData = audioBuffer.getChannelData(0);
      const frameStep = Math.max(1, Math.floor(audioBuffer.sampleRate * 0.015));
      const frameDuration = frameStep / audioBuffer.sampleRate;
      const spectrogram = tf.tidy(() => tf.abs(tf.signal.stft(tf.tensor1d(rawData), 256, frameStep)));
      const spectrogramData = await spectrogram.array() as number[][];
      spectrogram.dispose();
      const numFrames = spectrogramData.length;
      if (numFrames < detector.getWindowSize()) throw new Error("Audio za krótkie");
      const allScores: number[] = [];
      const points: any[] = [];
      for (let i = 0; i <= numFrames - detector.getWindowSize(); i++) {
        const sequence = spectrogramData.slice(i, i + detector.getWindowSize()).map(f => detector.scaleFrame(f.slice(0, 128)));
        const { score } = await detector.predict(sequence);
        const lastFrame = spectrogramData[i + detector.getWindowSize() - 1];
        const hz = lastFrame.reduce((a, b, idx) => a + (b * idx * 172), 0) / (lastFrame.reduce((a, b) => a + b, 0) || 1);
        allScores.push(score);
        points.push({ timestamp: (i + detector.getWindowSize()) * frameDuration, hz, score });
        if (i % 60 === 0) { setBatchProgress(Math.round((i / numFrames) * 100)); await tf.nextFrame(); }
      }
      const finalThresh = detector.calculateRobustThreshold(allScores);
      setDynamicThreshold(finalThresh);
      const processedChart: AudioChartData[] = [];
      const foundAnoms: Anomaly[] = [];
      let smoothScore = 0;
      points.forEach((p, idx) => {
        smoothScore = smoothScore * 0.8 + p.score * 0.2;
        if (idx % 2 === 0) processedChart.push({ time: p.timestamp.toFixed(2), amplitude: p.hz, anomalyLevel: smoothScore, second: p.timestamp });
        if (smoothScore > finalThresh) {
          const last = foundAnoms[foundAnoms.length - 1];
          if (!last || (p.timestamp - (last.offsetSeconds! + last.durationSeconds)) > 0.4) {
             foundAnoms.push({ id: `a-${idx}`, timestamp: new Date(), offsetSeconds: p.timestamp, durationSeconds: frameDuration, intensity: smoothScore/finalThresh, severity: smoothScore > finalThresh*2 ? 'High' : 'Medium', description: 'Wykryto odchyłkę', type: 'Audio Sentinel' });
          } else { last.durationSeconds = p.timestamp - last.offsetSeconds!; last.intensity = Math.max(last.intensity, smoothScore/finalThresh); }
        }
      });
      setChartData(processedChart);
      setAnomalies(foundAnoms.filter(a => a.durationSeconds > 0.15));
      addLog(`Analiza ${file.name} gotowa`, "success");
      setMode('IDLE');
    } catch (err: any) { addLog(`Błąd: ${err.message}`, "error"); setMode('IDLE'); }
  };

  const startLive = async () => {
    try {
      setMode('MONITORING');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const audioCtx = new AudioContext();
      audioContextRef.current = audioCtx;
      const analyzer = audioCtx.createAnalyser();
      analyzer.fftSize = 256;
      audioCtx.createMediaStreamSource(stream).connect(analyzer);
      monitoringRef.current = true;
      const loop = async () => {
        if (!monitoringRef.current) return;
        const data = new Float32Array(analyzer.frequencyBinCount);
        analyzer.getFloatFrequencyData(data);
        const scaled = detector.scaleFrame(Array.from(data).map(v => Math.pow(10, v/20) * 1000));
        liveBufferRef.current.push(scaled);
        if (liveBufferRef.current.length > detector.getWindowSize()) liveBufferRef.current.shift();
        if (liveBufferRef.current.length === detector.getWindowSize()) {
          const { score } = await detector.predict(liveBufferRef.current);
          setCurrentScore(score);
          setChartData(p => [...p.slice(-100), { time: 'Now', amplitude: 2000, anomalyLevel: score, second: Date.now() }]);
        }
        requestAnimationFrame(loop);
      };
      loop();
    } catch (e) { addLog("Błąd mikrofonu", "error"); setMode('IDLE'); }
  };

  const seekTo = (s: number) => { if (mainAudioRef.current) { mainAudioRef.current.currentTime = s; mainAudioRef.current.play().catch(()=>{}); } };

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100 p-4 lg:p-6 font-sans overflow-x-hidden">
      <header className="flex flex-col xl:flex-row items-center justify-between gap-6 mb-8 w-full">
          <div className="flex items-center gap-4">
            <div className="bg-indigo-600 p-3 rounded-2xl shadow-xl animate-glow"><BrainCircuit className="w-8 h-8" /></div>
            <div>
              <h1 className="text-2xl font-black italic uppercase leading-none tracking-tighter">AUDIO<span className="text-indigo-400">SENTINEL</span> <span className="text-[10px] align-top text-slate-500">v1.5</span></h1>
              <div className="flex gap-2 mt-2">
                   <div className="flex items-center gap-1.5 bg-slate-900 px-2 py-1 rounded-lg border border-slate-800">
                      <span className={`w-1.5 h-1.5 rounded-full ${mode !== 'IDLE' ? 'bg-emerald-500 animate-pulse' : 'bg-slate-700'}`}></span>
                      <span className="text-[9px] font-black uppercase text-slate-400">{status}</span>
                   </div>
                   <div className="flex items-center gap-1.5 bg-indigo-500/10 px-2 py-1 rounded-lg border border-indigo-500/20 text-indigo-400 text-[9px] font-black uppercase">
                      <Cpu className="w-3 h-3" /> {tfBackend}
                   </div>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-3 bg-slate-900/60 p-3 rounded-2xl border border-slate-800 shadow-2xl backdrop-blur-xl">
             <div className="flex items-center gap-2 bg-slate-800/50 p-2 rounded-xl border border-slate-700/50">
                  <button className="relative bg-emerald-600/10 hover:bg-emerald-600 text-emerald-500 hover:text-white px-3 py-1.5 rounded-lg text-[10px] font-black uppercase transition-all border border-emerald-500/30">
                    <Database className="w-4 h-4 mr-2 inline" /> Normalny
                    <input type="file" multiple className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => {
                      const f = e.target.files;
                      if (f) setTrainingQueue(prev => [...prev, ...Array.from(f).map(x => ({ file: x, label: 'Normal' as const, status: 'pending' as const }))]);
                    }} />
                  </button>
                  <button className="relative bg-red-600/10 hover:bg-red-600 text-red-500 hover:text-white px-3 py-1.5 rounded-lg text-[10px] font-black uppercase transition-all border border-red-500/30">
                    <Database className="w-4 h-4 mr-2 inline" /> Anomalia
                    <input type="file" multiple className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => {
                      const f = e.target.files;
                      if (f) setTrainingQueue(prev => [...prev, ...Array.from(f).map(x => ({ file: x, label: 'Anomaly' as const, status: 'pending' as const }))]);
                    }} />
                  </button>
                  <button disabled={trainingQueue.length === 0 || mode !== 'IDLE'} onClick={runIncrementalTraining} className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 p-2 rounded-lg transition-all shadow-lg"><RefreshCw className={`w-5 h-5 ${mode === 'TRAINING' ? 'animate-spin' : ''}`} /></button>
             </div>

             <div className="flex items-center gap-2">
                <button disabled={mode !== 'IDLE'} className="relative bg-slate-800 hover:bg-slate-700 text-indigo-400 px-4 py-3 rounded-xl text-[10px] font-black uppercase border border-slate-700 flex items-center gap-2 transition-all">
                  <FileAudio className="w-4 h-4" /> Analizuj WAV
                  <input type="file" accept="audio/*" className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => e.target.files?.[0] && analyzeSingleFile(e.target.files[0])} />
                </button>
                {mode !== 'MONITORING' ? (
                  <button disabled={mode !== 'IDLE'} onClick={startLive} className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-3 rounded-xl text-[10px] font-black uppercase flex items-center gap-2 shadow-lg shadow-emerald-500/20"><Mic className="w-4 h-4" /> Live</button>
                ) : (
                  <button onClick={() => { monitoringRef.current = false; setMode('IDLE'); }} className="bg-red-600 hover:bg-red-500 text-white px-4 py-3 rounded-xl text-[10px] font-black uppercase flex items-center gap-2"><Square className="w-4 h-4" /> Stop</button>
                )}
             </div>
             <button onClick={() => setShowReportModal(true)} className="p-3 bg-indigo-600/20 hover:bg-indigo-600 text-indigo-400 hover:text-white rounded-xl transition-all border border-indigo-500/30 shadow-lg"><FileText className="w-5 h-5" /></button>
             <button onClick={() => setShowSettings(true)} className="p-3 bg-slate-800 hover:bg-slate-700 text-slate-400 rounded-xl transition-all border border-slate-700"><Settings className="w-5 h-5" /></button>
          </div>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 pb-20">
          <div className="lg:col-span-3 space-y-6">
             <div className="bg-slate-900/40 border border-slate-800/50 rounded-3xl p-6 shadow-2xl backdrop-blur-md">
                <h2 className="text-[10px] font-black text-indigo-400 uppercase tracking-widest flex items-center gap-2 mb-4"><BarChart3 className="w-4 h-4" /> Silnik</h2>
                <div className="grid grid-cols-2 gap-4">
                   <div className="bg-slate-950/40 p-3 rounded-xl border border-slate-800">
                      <p className="text-[8px] font-black text-slate-500 uppercase">Wzorce AI</p>
                      <p className="text-lg font-black text-white">{totalTrained}</p>
                   </div>
                   <div className="bg-slate-950/40 p-3 rounded-xl border border-slate-800">
                      <p className="text-[8px] font-black text-slate-500 uppercase">VRAM</p>
                      <p className="text-lg font-black text-emerald-400">{(ramUsage / 1024 / 1024).toFixed(1)}MB</p>
                   </div>
                </div>
                <div className="mt-6 pt-6 border-t border-slate-800">
                   <div className="flex items-center justify-between mb-2 text-[10px] font-black uppercase text-slate-400 tracking-tight">
                      <span>Czułość Sensora</span>
                      <span className="font-mono text-indigo-400">{sensitivity}</span>
                   </div>
                   <input type="range" min="1" max="10" step="0.5" value={sensitivity} onChange={(e) => { const v = parseFloat(e.target.value); setSensitivity(v); detector.setSensitivity(v); setDynamicThreshold(detector.getThreshold()); }} className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
                </div>
             </div>
             
             <div className="bg-slate-900/40 border border-slate-800/50 rounded-3xl p-6 h-[300px] flex flex-col shadow-inner">
                <h2 className="text-xs font-black text-slate-400 uppercase tracking-widest flex items-center gap-2 mb-4"><History className="w-4 h-4" /> Kolejka ({trainingQueue.length})</h2>
                <div className="flex-1 overflow-y-auto space-y-1 scrollbar-thin text-[9px] uppercase font-mono">
                  {trainingQueue.map((item, idx) => (
                    <div key={idx} className="bg-slate-950/30 border border-slate-800/50 p-2 rounded-lg flex items-center justify-between group">
                      <p className="truncate text-slate-400 pr-2">{item.file.name}</p>
                      <button onClick={() => setTrainingQueue(q => q.filter((_, i) => i !== idx))} className="text-red-500 opacity-0 group-hover:opacity-100 transition-all"><X className="w-3 h-3" /></button>
                    </div>
                  ))}
                </div>
             </div>
          </div>

          <div className="lg:col-span-6 flex flex-col gap-6">
             <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl">
                  <p className="text-slate-500 text-[9px] uppercase font-black mb-1">Status</p>
                  <span className={`text-xl font-mono font-black ${mode !== 'IDLE' ? 'text-indigo-400 animate-pulse' : 'text-emerald-500'}`}>{mode === 'IDLE' ? 'GOTOWY' : mode}</span>
                </div>
                <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl">
                  <p className="text-slate-500 text-[9px] uppercase font-black mb-1">Próg</p>
                  <p className="text-xl font-black text-white">{dynamicThreshold.toFixed(2)}</p>
                </div>
                <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl">
                  <p className="text-slate-500 text-[9px] uppercase font-black mb-1">Alertów</p>
                  <p className="text-xl font-black text-red-500">{anomalies.length}</p>
                </div>
                <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl">
                  <p className="text-slate-500 text-[9px] uppercase font-black mb-1">Engine</p>
                  <p className="text-xl font-black text-indigo-500">{tfBackend}</p>
                </div>
             </div>

             <div className="flex-1 bg-slate-900/40 border border-slate-800/50 rounded-[3rem] p-4 sm:p-8 flex flex-col shadow-2xl backdrop-blur-sm min-h-[450px]">
               {analyzedFileUrl && (
                  <div className="mb-6 bg-slate-950/80 p-3 rounded-2xl border border-slate-800 flex items-center gap-4 shadow-xl">
                     <Volume2 className="text-indigo-400 w-5 h-5" />
                     <audio ref={mainAudioRef} src={analyzedFileUrl} controls onTimeUpdate={() => setCurrentTime(mainAudioRef.current?.currentTime || 0)} className="flex-1 h-8 accent-indigo-500" />
                  </div>
               )}
               <div className="flex-1">
                  <AnomalyChart data={chartData} threshold={dynamicThreshold} anomalies={anomalies} currentTime={currentTime} onPointClick={(p) => p.second !== undefined && seekTo(p.second)} />
               </div>
             </div>
          </div>

          <div className="lg:col-span-3 flex flex-col gap-6">
             <div className="bg-slate-900/40 border border-slate-800/50 rounded-3xl p-6 h-[450px] flex flex-col shadow-xl">
                <h2 className="text-xs font-black text-slate-400 uppercase tracking-widest flex items-center gap-2 mb-6"><History className="w-4 h-4 text-indigo-500" /> Log Zdarzeń</h2>
                <div className="flex-1 overflow-y-auto space-y-4 pr-1 scrollbar-thin">
                  {anomalies.map((a) => (
                    <div key={a.id} className="p-3 rounded-2xl border border-slate-800 bg-slate-950/60 hover:border-indigo-500/60 cursor-pointer transition-all active:scale-95" onClick={() => seekTo(a.offsetSeconds || 0)}>
                      <div className="flex justify-between items-center mb-1">
                          <span className={`text-[8px] font-black uppercase px-1 rounded text-white ${a.severity === 'High' ? 'bg-red-500' : 'bg-amber-500'}`}>{a.severity}</span>
                          <span className="text-[10px] font-mono text-slate-500">{a.offsetSeconds?.toFixed(2)}s</span>
                      </div>
                      <p className="text-[10px] text-slate-400 font-bold uppercase tracking-tighter">Odchyłka: x{(a.intensity).toFixed(1)}</p>
                    </div>
                  ))}
                  {anomalies.length === 0 && <div className="flex flex-col items-center justify-center h-full opacity-20"><Zap className="w-12 h-12 mb-4" /><p className="text-xs uppercase font-black">Brak wykryć</p></div>}
                </div>
             </div>

             <div className="bg-slate-900/40 border border-slate-800/50 rounded-3xl p-4 flex flex-col">
                <button onClick={() => setShowConsole(!showConsole)} className="flex items-center justify-between w-full text-[10px] font-black uppercase text-slate-500 mb-2 hover:text-indigo-400 transition-all">
                  <div className="flex items-center gap-2"><Terminal className="w-3.5 h-3.5" /> Konsola Engine</div>
                  {showConsole ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                </button>
                {showConsole && (
                  <div className="h-32 bg-black/40 rounded-xl p-3 font-mono text-[9px] overflow-y-auto scrollbar-thin border border-slate-800/50">
                    {logs.map((log, i) => <div key={i} className={`mb-1 ${log.type === 'error' ? 'text-red-400' : log.type === 'success' ? 'text-emerald-400' : 'text-slate-400'}`}>[{log.time}] {log.message}</div>)}
                  </div>
                )}
             </div>
          </div>
      </main>

      {showSettings && (
        <div className="fixed inset-0 bg-slate-950/90 backdrop-blur-xl z-[500] flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-[2.5rem] w-full max-w-lg p-8 shadow-2xl overflow-hidden relative">
            <div className="absolute top-0 left-0 w-full h-1 bg-indigo-500 shadow-[0_0_20px_rgba(79,70,229,0.5)]"></div>
            <h2 className="text-xl font-black uppercase text-white mb-8 flex items-center gap-3"><Settings className="w-6 h-6 text-indigo-400" /> Zarządzanie Wiedzą AI</h2>
            
            <div className="space-y-4">
               <div className="p-4 bg-slate-950/50 rounded-2xl border border-slate-800">
                  <h3 className="text-[10px] font-black uppercase text-slate-500 mb-4 tracking-widest">Kopia Zapasowa (Plik)</h3>
                  <button onClick={() => detector.exportModel()} className="w-full bg-indigo-600 p-3 rounded-xl font-black uppercase text-white hover:bg-indigo-500 transition-all flex items-center justify-center gap-2 shadow-lg">
                    <Download className="w-4 h-4" /> Pobierz Bazę AI
                  </button>
                  <p className="text-[9px] text-slate-600 mt-2 italic text-center">Pobierze 3 pliki: model.json, weights.bin oraz stats.json</p>
               </div>

               <div className="p-4 bg-slate-950/50 rounded-2xl border border-slate-800">
                  <h3 className="text-[10px] font-black uppercase text-slate-500 mb-4 tracking-widest">Importuj z Dysku</h3>
                  <div className="grid grid-cols-1 gap-2 mb-4">
                    <div className="flex flex-col gap-1">
                      <span className="text-[8px] uppercase font-bold text-slate-500">1. model.json</span>
                      <input type="file" ref={importJsonRef} className="text-[10px] text-slate-400 bg-slate-900 p-1 rounded border border-slate-800" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <span className="text-[8px] uppercase font-bold text-slate-500">2. weights.bin</span>
                      <input type="file" ref={importWeightsRef} className="text-[10px] text-slate-400 bg-slate-900 p-1 rounded border border-slate-800" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <span className="text-[8px] uppercase font-bold text-slate-500">3. stats.json (opcjonalnie)</span>
                      <input type="file" ref={importStatsRef} className="text-[10px] text-slate-400 bg-slate-900 p-1 rounded border border-slate-800" />
                    </div>
                  </div>
                  <button onClick={handleImport} className="w-full bg-emerald-600/20 text-emerald-500 border border-emerald-500/30 p-3 rounded-xl font-black uppercase hover:bg-emerald-600 hover:text-white transition-all flex items-center justify-center gap-2">
                    <FileUp className="w-4 h-4" /> Wgraj do Pamięci
                  </button>
               </div>

               <button onClick={async () => { if(window.confirm("Zresetować wszystko?")) { await detector.clearKnowledge(); setTotalTrained(0); setShowSettings(false); addLog("Baza wiedzy zresetowana", "warning"); } }} className="w-full bg-red-600/10 text-red-500 border border-red-500/20 p-3 rounded-xl font-black uppercase hover:bg-red-600 hover:text-white transition-all flex items-center justify-center gap-2">
                 <Trash2 className="w-4 h-4" /> Usuń Lokalną Wiedzę
               </button>
            </div>

            <button onClick={() => setShowSettings(false)} className="w-full text-slate-500 font-black uppercase py-4 text-xs mt-4 hover:text-white transition-all underline decoration-slate-800 underline-offset-8">Zamknij</button>
          </div>
        </div>
      )}

      {showReportModal && (
        <div className="fixed inset-0 bg-slate-950/95 backdrop-blur-xl z-[1000] flex flex-col p-4 sm:p-8 overflow-y-auto">
          <div className="max-w-[210mm] mx-auto w-full flex flex-col gap-6">
            <div className="flex justify-between items-center">
              <button onClick={() => setShowReportModal(false)} className="flex items-center gap-2 text-slate-400 hover:text-white transition-all uppercase text-[10px] font-black group"><X className="w-5 h-5 group-hover:rotate-90 transition-all" /> Zamknij</button>
              <button onClick={() => window.print()} className="bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-3 rounded-xl flex items-center gap-2 uppercase text-xs font-black shadow-2xl transition-all active:scale-95"><Printer className="w-5 h-5" /> Generuj Dokument PDF</button>
            </div>
            <TechnicalReportView anomalies={anomalies} totalTrained={totalTrained} sensitivity={sensitivity} threshold={dynamicThreshold} />
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
