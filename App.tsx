
import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, Square, BrainCircuit, Loader2, 
  Settings, Upload, FileAudio, History, Download, 
  Zap, Volume2, FileSearch, Mic, Trash2, 
  ChevronDown, ChevronRight, Terminal, Database,
  SlidersHorizontal, Save, RefreshCw, BarChart3, Cpu, AlertTriangle,
  HardDrive, FolderOpen, Sliders
} from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import { Anomaly, AudioChartData, ModelConfig } from './types';
import AnomalyChart from './components/AnomalyChart';
import ReportTable from './components/ReportTable';
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
  const [trainingQueue, setTrainingQueue] = useState<FileQueueItem[]>([]);
  const [batchProgress, setBatchProgress] = useState(0);
  const [currentScore, setCurrentScore] = useState(0);
  const [analyzedFileUrl, setAnalyzedFileUrl] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [totalTrained, setTotalTrained] = useState(0);
  const [isPersistent, setIsPersistent] = useState(false);
  const [ramUsage, setRamUsage] = useState(0);
  const [sensitivity, setSensitivity] = useState(detector.getSensitivity());
  const [dynamicThreshold, setDynamicThreshold] = useState(detector.getThreshold());
  
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [showConsole, setShowConsole] = useState(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mainAudioRef = useRef<HTMLAudioElement | null>(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const monitoringRef = useRef<boolean>(false);
  const liveBufferRef = useRef<number[][]>([]);

  useEffect(() => {
    let memInterval: any;
    const checkStatus = async () => {
      try {
        await tf.ready();
        setTfBackend(detector.getBackendName());
        setIsPersistent(detector.getIsLoadedFromStorage());
        setTotalTrained(detector.getTotalFiles());
        setStatus('System gotowy');
        memInterval = setInterval(() => setRamUsage(detector.getMemInfo()?.numBytes || 0), 5000);
      } catch (err) {
        setStatus('Błąd TF');
      }
    };
    checkStatus();
    return () => clearInterval(memInterval);
  }, []);

  const addLog = (message: string, type: LogEntry['type'] = 'info') => {
    setLogs(prev => [...prev.slice(-49), {
      message, type,
      time: new Date().toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' })
    }]);
  };

  const seekTo = (seconds: number) => {
    if (mainAudioRef.current) {
      mainAudioRef.current.currentTime = seconds;
      mainAudioRef.current.play().catch(() => {});
    }
  };

  const addToQueue = (files: FileList | null, label: 'Normal' | 'Anomaly') => {
    if (!files) return;
    const newItems: FileQueueItem[] = Array.from(files)
      .filter(f => f.type.startsWith('audio/'))
      .map(f => ({ file: f, label, status: 'pending' }));
    setTrainingQueue(prev => [...prev, ...newItems]);
    addLog(`Dodano ${newItems.length} plików (${label})`);
  };

  const calculateCentroid = (magnitudes: number[], sampleRate: number): number => {
    const binHz = (sampleRate / 2) / magnitudes.length;
    let numerator = 0;
    let denominator = 0;
    for (let i = 0; i < magnitudes.length; i++) {
      // Skupiamy się na słyszalnym paśmie maszyny (np. do 8kHz)
      const weight = magnitudes[i];
      numerator += (i * binHz) * weight;
      denominator += weight;
    }
    return denominator < 0.0001 ? 0 : numerator / denominator;
  };

  const runIncrementalTraining = async () => {
    if (trainingQueue.length === 0) return;
    setMode('TRAINING');
    const audioCtx = new AudioContext();
    for (let i = 0; i < trainingQueue.length; i++) {
      const item = trainingQueue[i];
      setBatchProgress(Math.round((i / trainingQueue.length) * 100));
      try {
        const arrayBuffer = await item.file.arrayBuffer();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const channelData = audioBuffer.getChannelData(0);
        const frameStep = Math.floor(audioBuffer.sampleRate * 0.015);
        const audioTensor = tf.tensor1d(channelData);
        const spectrogram = tf.signal.stft(audioTensor, 256, frameStep);
        const magnitudes = tf.abs(spectrogram);
        const spectrogramData = await magnitudes.array() as number[][];
        const frames = spectrogramData.map(row => detector.scaleFrame(row.slice(0, 128)));
        const { error } = await detector.trainOnFile(frames, item.label);
        audioTensor.dispose(); spectrogram.dispose(); magnitudes.dispose();
        addLog(`Uczono: ${item.file.name} (E: ${error.toFixed(2)})`);
        setTotalTrained(detector.getTotalFiles());
        setIsPersistent(true);
      } catch (err) {
        addLog(`Błąd pliku: ${item.file.name}`, 'error');
      }
      await tf.nextFrame();
    }
    setTrainingQueue([]);
    setMode('IDLE');
    setStatus('Gotowy');
  };

  const analyzeSingleFile = async (file: File) => {
    try {
      setMode('FILE_ANALYSIS');
      setStatus('Skanowanie...');
      setChartData([]);
      setAnomalies([]);
      const fileUrl = URL.createObjectURL(file);
      setAnalyzedFileUrl(fileUrl);
      const audioCtx = new AudioContext();
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const channelData = audioBuffer.getChannelData(0);
      
      const frameStep = Math.floor(audioBuffer.sampleRate * 0.015);
      const frameDurationSec = frameStep / audioBuffer.sampleRate;
      const windowSize = detector.getWindowSize();
      
      const audioTensor = tf.tensor1d(channelData);
      const spectrogram = tf.signal.stft(audioTensor, 256, frameStep);
      const magnitudes = tf.abs(spectrogram);
      const spectrogramData = await magnitudes.array() as number[][];
      const numFrames = spectrogramData.length;
      
      const allScores: number[] = [];
      const preliminaryData: {timestamp: number, hz: number, score: number}[] = [];

      // Faza 1: Obliczanie błędów rekonstrukcji
      for (let i = 0; i <= numFrames - windowSize; i++) {
        const rawSequence = spectrogramData.slice(i, i + windowSize);
        const sequence = rawSequence.map(frame => detector.scaleFrame(frame.slice(0, 128)));
        const { score } = await detector.predict(sequence);
        const hz = calculateCentroid(rawSequence[windowSize-1].slice(0, 128), audioBuffer.sampleRate);
        
        allScores.push(score);
        preliminaryData.push({ timestamp: (i + windowSize) * frameDurationSec, hz, score });
        
        if (i % 600 === 0) {
          setBatchProgress(Math.round((i / numFrames) * 50));
          await tf.nextFrame();
        }
      }

      // Faza 2: Detekcja charakterystycznych wychyleń (Robust MAD)
      const localThresh = detector.calculateRobustThreshold(allScores);
      setDynamicThreshold(localThresh);
      addLog(`Auto-kalibracja zakończona (Próg: ${localThresh.toFixed(2)})`, 'success');

      const finalChartData: AudioChartData[] = [];
      const finalAnomalies: Anomaly[] = [];
      let smoothScoreBuffer: number[] = [];
      let smoothHzBuffer: number[] = [];

      preliminaryData.forEach((point, idx) => {
        smoothScoreBuffer.push(point.score);
        if (smoothScoreBuffer.length > 10) smoothScoreBuffer.shift();
        const smoothedScore = smoothScoreBuffer.reduce((a,b)=>a+b,0) / smoothScoreBuffer.length;

        smoothHzBuffer.push(point.hz);
        if (smoothHzBuffer.length > 5) smoothHzBuffer.shift();
        const smoothedHz = smoothHzBuffer.reduce((a,b)=>a+b,0) / smoothHzBuffer.length;

        if (idx % 2 === 0) {
          finalChartData.push({
            time: point.timestamp.toFixed(2),
            amplitude: smoothedHz,
            anomalyLevel: smoothedScore,
            second: point.timestamp
          });
        }

        if (smoothedScore > localThresh) {
          const lastAnom = finalAnomalies[finalAnomalies.length - 1];
          // Dodano warunek "characteristic delay" aby uniknąć glichy
          if (!lastAnom || (point.timestamp - (lastAnom.offsetSeconds! + lastAnom.durationSeconds)) > 0.6) {
            finalAnomalies.push({ 
              id: `anom-${idx}`, 
              timestamp: new Date(), 
              offsetSeconds: point.timestamp, 
              durationSeconds: frameDurationSec * 6, 
              intensity: smoothedScore / localThresh, 
              severity: smoothedScore > localThresh * 2.2 ? 'High' : 'Medium', 
              description: `ZDARZENIE`, 
              type: 'Acoustic Shift' 
            });
          } else {
            lastAnom.durationSeconds = point.timestamp - lastAnom.offsetSeconds!;
            lastAnom.intensity = Math.max(lastAnom.intensity, smoothedScore / localThresh);
          }
        }
        
        if (idx % 1000 === 0) setBatchProgress(50 + Math.round((idx / preliminaryData.length) * 50));
      });

      setChartData(finalChartData);
      setAnomalies(finalAnomalies);
      setMode('IDLE');
      setStatus('Gotowy');
      audioTensor.dispose(); spectrogram.dispose(); magnitudes.dispose();
      addLog(`Wykryto ${finalAnomalies.length} istotnych anomalii powyżej tła.`, finalAnomalies.length > 0 ? 'warning' : 'success');
    } catch (err) {
      addLog("Błąd analizy", "error");
      setMode('IDLE');
    }
  };

  const startLive = async () => {
    try {
      setMode('MONITORING');
      setStatus('LIVE');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const audioCtx = new AudioContext();
      audioContextRef.current = audioCtx;
      const analyzer = audioCtx.createAnalyser();
      analyzer.fftSize = 256;
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyzer);
      monitoringRef.current = true;
      const loop = async () => {
        if (!monitoringRef.current) return;
        const data = new Float32Array(analyzer.frequencyBinCount);
        analyzer.getFloatFrequencyData(data);
        const scaledFrame = detector.scaleFrame(Array.from(data).map(v => Math.pow(10, v/20) * 1000));
        liveBufferRef.current.push(scaledFrame);
        if (liveBufferRef.current.length > detector.getWindowSize()) liveBufferRef.current.shift();
        if (liveBufferRef.current.length === detector.getWindowSize()) {
          const { score } = await detector.predict(liveBufferRef.current);
          setCurrentScore(score);
          setChartData(prev => [...prev.slice(-99), { 
            time: new Date().toLocaleTimeString([], {second:'2-digit'}), 
            amplitude: calculateCentroid(Array.from(data).map(v => Math.pow(10, v/20) * 1000), 44100), 
            anomalyLevel: score, 
            second: Date.now() / 1000 
          }]);
        }
        requestAnimationFrame(loop);
      };
      loop();
    } catch (err) {
      addLog("Błąd mikrofonu", "error");
      setMode('IDLE');
    }
  };

  const resetModel = async () => {
    if (window.confirm("Zresetować bazę AI?")) {
      await detector.clearKnowledge();
      setTotalTrained(0);
      setIsPersistent(false);
      setAnomalies([]);
      setChartData([]);
      addLog("Baza zresetowana", "warning");
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100 p-2 sm:p-4 lg:p-6 font-sans overflow-x-hidden">
      <header className="flex flex-col xl:flex-row items-center justify-between gap-6 mb-8 w-full">
        <div className="flex items-center gap-4 w-full xl:w-auto">
          <div className="bg-indigo-600 p-3 rounded-2xl shadow-xl animate-glow">
            <BrainCircuit className="w-8 h-8" />
          </div>
          <div className="flex-1">
            <h1 className="text-2xl sm:text-3xl font-black tracking-tighter italic uppercase leading-none">AUDIO<span className="text-indigo-400">SENTINEL</span></h1>
            <div className="flex flex-wrap items-center gap-2 mt-2">
                 <div className="flex items-center gap-1.5 bg-slate-900/80 px-2 py-1 rounded-lg border border-slate-800">
                    <span className={`w-2 h-2 rounded-full ${mode !== 'IDLE' ? 'bg-emerald-500 animate-pulse' : 'bg-slate-700'}`}></span>
                    <span className="text-[10px] font-black text-slate-300 uppercase tracking-widest">{status}</span>
                 </div>
                 <div className="flex items-center gap-1.5 bg-indigo-500/10 px-2 py-1 rounded-lg border border-indigo-500/20">
                    <Cpu className="w-3 h-3 text-indigo-400" />
                    <span className="text-[10px] font-black uppercase text-indigo-400">{tfBackend}</span>
                 </div>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-3 bg-slate-900/60 p-2 sm:p-3 rounded-2xl border border-slate-800 shadow-2xl backdrop-blur-xl w-full xl:w-auto">
           <div className="flex items-center gap-2 bg-slate-800/50 p-2 rounded-2xl border border-slate-700/50">
             <div className="flex flex-col gap-1">
                <button className="relative bg-emerald-600/10 hover:bg-emerald-600 text-emerald-500 hover:text-white px-3 py-1.5 rounded-lg text-[9px] font-black uppercase transition-all flex items-center gap-2 border border-emerald-500/30">
                  <Database className="w-3.5 h-3.5" /> Normal
                  <input type="file" multiple className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => addToQueue(e.target.files, 'Normal')} />
                </button>
                <button className="relative bg-red-600/10 hover:bg-red-600 text-red-500 hover:text-white px-3 py-1.5 rounded-lg text-[9px] font-black uppercase transition-all flex items-center gap-2 border border-red-500/30">
                  <Database className="w-3.5 h-3.5" /> Anomalia
                  <input type="file" multiple className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => addToQueue(e.target.files, 'Anomaly')} />
                </button>
             </div>
             <button 
                disabled={mode === 'TRAINING' || trainingQueue.length === 0} 
                onClick={runIncrementalTraining} 
                className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 text-white px-4 py-3 rounded-xl transition-all shadow-lg flex flex-col items-center justify-center gap-1 min-w-[100px]"
             >
                {mode === 'TRAINING' ? <Loader2 className="w-5 h-5 animate-spin" /> : <RefreshCw className="w-5 h-5" />}
                <span className="text-[9px] font-black uppercase">Trenuj</span>
             </button>
           </div>

           <div className="flex items-center gap-2 sm:gap-3">
              <button className="relative bg-slate-800 hover:bg-slate-700 text-indigo-400 px-4 py-3 rounded-xl text-[10px] font-black uppercase border border-slate-700 flex items-center gap-2">
                <FileAudio className="w-4 h-4" /> Skanuj Audio
                <input type="file" accept="audio/*" className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => e.target.files?.[0] && analyzeSingleFile(e.target.files[0])} />
              </button>
              {mode !== 'MONITORING' ? (
                <button onClick={startLive} className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-3 rounded-xl text-[10px] font-black uppercase shadow-lg shadow-emerald-500/20 flex items-center gap-2"><Mic className="w-4 h-4" /> Live</button>
              ) : (
                <button onClick={() => { monitoringRef.current = false; setMode('IDLE'); }} className="bg-red-600 hover:bg-red-500 text-white px-4 py-3 rounded-xl text-[10px] font-black uppercase flex items-center gap-2"><Square className="w-4 h-4" /> Stop</button>
              )}
           </div>

           <button onClick={() => setShowSettings(true)} className="p-3 bg-slate-800 hover:bg-slate-700 text-slate-400 rounded-xl transition-all border border-slate-700">
             <Settings className="w-5 h-5" />
           </button>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 pb-20">
        <div className="lg:col-span-3 space-y-6">
           <div className="bg-slate-900/40 border border-slate-800/50 rounded-3xl p-6 shadow-2xl backdrop-blur-md">
              <h2 className="text-[10px] font-black text-indigo-400 uppercase tracking-widest flex items-center gap-2 mb-4"><BarChart3 className="w-4 h-4" /> Diagnostyka</h2>
              <div className="grid grid-cols-2 gap-4">
                 <div className="bg-slate-950/40 p-3 rounded-xl border border-slate-800">
                    <p className="text-[8px] font-black text-slate-500 uppercase">Nauczone</p>
                    <p className="text-lg font-black text-white">{totalTrained}</p>
                 </div>
                 <div className="bg-slate-950/40 p-3 rounded-xl border border-slate-800">
                    <p className="text-[8px] font-black text-slate-500 uppercase">Próg Błędu</p>
                    <p className="text-lg font-black text-emerald-400">{dynamicThreshold.toFixed(2)}</p>
                 </div>
              </div>
              <div className="mt-6 pt-6 border-t border-slate-800">
                 <div className="flex items-center justify-between mb-2">
                    <p className="text-[9px] font-black text-slate-400 uppercase flex items-center gap-1.5"><Sliders className="w-3.5 h-3.5" /> Czułość</p>
                    <span className="text-[10px] font-mono text-indigo-400 font-black">{sensitivity}</span>
                 </div>
                 <input 
                    type="range" min="1" max="10" step="0.5" 
                    value={sensitivity} 
                    onChange={(e) => {
                       const val = parseFloat(e.target.value);
                       setSensitivity(val);
                       detector.setSensitivity(val);
                       setDynamicThreshold(detector.getThreshold());
                    }}
                    className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500" 
                 />
              </div>
           </div>
           
           <div className="bg-slate-900/40 border border-slate-800/50 rounded-3xl p-6 h-[300px] flex flex-col">
              <h2 className="text-xs font-black text-slate-400 uppercase tracking-widest flex items-center gap-2 mb-4"><Upload className="w-4 h-4 text-indigo-500" /> Kolejka ({trainingQueue.length})</h2>
              <div className="flex-1 overflow-y-auto space-y-1 pr-1 scrollbar-thin">
                {trainingQueue.map((item, idx) => (
                  <div key={idx} className="bg-slate-950/30 border border-slate-800/50 p-2 rounded-lg flex items-center justify-between">
                    <p className="text-[9px] font-medium truncate text-slate-400 uppercase">{item.file.name}</p>
                    <button onClick={() => setTrainingQueue(q => q.filter((_, i) => i !== idx))} className="text-red-500"><Trash2 className="w-3 h-3" /></button>
                  </div>
                ))}
              </div>
           </div>
        </div>

        <div className="lg:col-span-6 flex flex-col gap-6">
           <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl">
                <p className="text-slate-500 text-[9px] uppercase font-black mb-1">Score</p>
                <span className={`text-xl font-mono font-black ${currentScore > dynamicThreshold ? 'text-red-500 animate-pulse' : 'text-indigo-400'}`}>
                   {mode === 'FILE_ANALYSIS' ? `${batchProgress}%` : currentScore.toFixed(2)}
                </span>
              </div>
              <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl">
                <p className="text-slate-500 text-[9px] uppercase font-black mb-1">Częstotliwość</p>
                <p className="text-xl font-black text-white">{chartData.length > 0 ? chartData[chartData.length-1].amplitude.toFixed(0) : '0'} Hz</p>
              </div>
              <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl">
                <p className="text-slate-500 text-[9px] uppercase font-black mb-1">Piki</p>
                <p className="text-xl font-black text-white">{anomalies.length}</p>
              </div>
              <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-xl">
                <p className="text-slate-500 text-[9px] uppercase font-black mb-1">Baza AI</p>
                <p className={`text-xl font-black ${isPersistent ? 'text-emerald-400' : 'text-slate-500'}`}>{isPersistent ? 'OK' : 'BRAK'}</p>
              </div>
           </div>

           <div className="flex-1 bg-slate-900/40 border border-slate-800/50 rounded-[3rem] p-4 sm:p-8 flex flex-col shadow-2xl backdrop-blur-sm min-h-[450px]">
             {analyzedFileUrl && (
                <div className="mb-6 bg-slate-950/80 p-3 rounded-2xl border border-slate-800 flex items-center gap-4">
                   <Volume2 className="text-indigo-400 w-5 h-5" />
                   <audio ref={mainAudioRef} src={analyzedFileUrl} controls onTimeUpdate={() => setCurrentTime(mainAudioRef.current?.currentTime || 0)} className="flex-1 h-8" />
                </div>
             )}
             <div className="flex-1">
                <AnomalyChart 
                  data={chartData} 
                  threshold={dynamicThreshold} 
                  anomalies={anomalies} 
                  currentTime={currentTime} 
                  onPointClick={(p) => p.second !== undefined && seekTo(p.second)} 
                />
             </div>
           </div>
        </div>

        <div className="lg:col-span-3 flex flex-col gap-6">
           <div className="bg-slate-900/40 border border-slate-800/50 rounded-3xl p-6 h-[450px] flex flex-col shadow-xl">
              <h2 className="text-xs font-black text-slate-400 uppercase tracking-widest flex items-center gap-2 mb-6"><History className="w-4 h-4 text-indigo-500" /> Wychylenia</h2>
              <div className="flex-1 overflow-y-auto space-y-4 pr-1 scrollbar-thin">
                {anomalies.map((a) => (
                  <div key={a.id} className="p-3 rounded-2xl border border-slate-800 bg-slate-950/60 hover:border-indigo-500/60 cursor-pointer" onClick={() => seekTo(a.offsetSeconds || 0)}>
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-[8px] font-black uppercase bg-red-500 px-1 rounded text-white">{a.severity}</span>
                        <span className="text-[10px] font-mono text-slate-500">{a.offsetSeconds?.toFixed(2)}s</span>
                    </div>
                    <p className="text-[10px] text-slate-400 font-bold italic">Skok: {(a.intensity).toFixed(1)}x ponad tło</p>
                  </div>
                ))}
              </div>
           </div>

           <div className="bg-slate-900/40 border border-slate-800/50 rounded-3xl p-4 flex flex-col">
              <button onClick={() => setShowConsole(!showConsole)} className="flex items-center justify-between w-full text-[10px] font-black uppercase text-slate-500">
                <div className="flex items-center gap-2"><Terminal className="w-3.5 h-3.5" /> Konsola Diagnostyczna</div>
                {showConsole ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              </button>
              {showConsole && (
                <div className="mt-3 h-32 bg-black/60 rounded-xl p-3 font-mono text-[9px] overflow-y-auto scrollbar-thin">
                  {logs.map((log, i) => <div key={i} className={`text-xs ${log.type === 'warning' ? 'text-amber-400' : 'text-indigo-300'}`}>{log.message}</div>)}
                </div>
              )}
           </div>
        </div>
      </main>

      {anomalies.length > 0 && mode === 'IDLE' && <section className="mt-8 pb-10"><ReportTable anomalies={anomalies} /></section>}

      {showSettings && (
        <div className="fixed inset-0 bg-slate-950/90 backdrop-blur-md z-[500] flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl w-full max-w-xl p-8 shadow-2xl">
            <h2 className="text-xl font-black uppercase text-white mb-8">Ustawienia Systemu</h2>
            <div className="space-y-6">
               <button onClick={() => detector.exportModel()} className="w-full bg-emerald-600 p-3 rounded-xl font-black uppercase text-white">Zapisz Model</button>
               <button onClick={resetModel} className="w-full border border-red-900 text-red-500 p-3 rounded-xl font-black uppercase">Usuń całą bazę AI</button>
               <button onClick={() => setShowSettings(false)} className="w-full text-slate-500 font-black uppercase py-2">Zamknij</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
