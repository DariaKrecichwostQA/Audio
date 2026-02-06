
import * as tf from '@tensorflow/tfjs';
import { ModelConfig } from '../types';

const STORAGE_PATH = 'indexeddb://sentinel-model-v1';

class AnomalyDetector {
  private model: tf.LayersModel | null = null;
  private config: ModelConfig = {
    epochs: 1, 
    learningRate: 0.001,
    batchSize: 8, 
    latentDim: 24
  };
  private threshold: number = 3.5; 
  private sensitivity: number = 5.0; 
  private windowSize: number = 12;
  private errorStats: number[] = [];
  private totalProcessedFiles: number = 0;
  private isModelLoaded: boolean = false;
  private isReady: boolean = false;

  constructor() {
    this.init();
  }

  private async init() {
    try {
      await tf.ready();
      try {
        await tf.setBackend('webgpu');
      } catch (e) {
        try {
          await tf.setBackend('webgl');
        } catch (e2) {
          await tf.setBackend('cpu');
        }
      }
      this.isReady = true;
      
      try {
        this.model = await tf.loadLayersModel(STORAGE_PATH);
        this.model.compile({
          optimizer: tf.train.adam(this.config.learningRate),
          loss: 'meanSquaredError'
        });
        this.isModelLoaded = true;
      } catch (loadErr) {
        await this.createNewModel();
      }
      this.loadStats();
    } catch (e) {
      console.error("AI Init Error:", e);
    }
  }

  private loadStats() {
    const savedStats = localStorage.getItem('sentinel_stats');
    if (savedStats) {
      const parsed = JSON.parse(savedStats);
      this.errorStats = parsed.errorStats || [];
      this.totalProcessedFiles = parsed.totalProcessedFiles || 0;
      this.threshold = parsed.threshold || 3.5;
      this.sensitivity = parsed.sensitivity || 5.0;
    }
  }

  private async createNewModel() {
    const input = tf.input({ shape: [this.windowSize, 128] });
    let encoder = tf.layers.lstm({ units: 64, returnSequences: true }).apply(input);
    encoder = tf.layers.lstm({ units: this.config.latentDim, returnSequences: false }).apply(encoder) as tf.SymbolicTensor;
    const repeat = tf.layers.repeatVector({ n: this.windowSize }).apply(encoder);
    let decoder = tf.layers.lstm({ units: 64, returnSequences: true }).apply(repeat);
    const output = tf.layers.timeDistributed({
      layer: tf.layers.dense({ units: 128, activation: 'sigmoid' })
    }).apply(decoder) as tf.SymbolicTensor;
    this.model = tf.model({ inputs: input, outputs: output });
    this.model.compile({ optimizer: tf.train.adam(this.config.learningRate), loss: 'meanSquaredError' });
  }

  public getThreshold() { return this.threshold; }
  public getSensitivity() { return this.sensitivity; }
  public setSensitivity(val: number) { 
    this.sensitivity = val; 
    this.updateThreshold(); 
    this.persist();
  }
  public getWindowSize() { return this.windowSize; }
  public getTotalFiles() { return this.totalProcessedFiles; }
  public getBackendName() { return this.isReady ? tf.getBackend().toUpperCase() : 'WAITING'; }
  public getIsLoadedFromStorage() { return this.isModelLoaded; }

  public scaleFrame(frame: number[]) {
    const len = frame.length;
    return frame.map((v, i) => {
      const boost = 1.0 + (i / len) * 1.5;
      const boostedVal = v * boost;
      const cleaned = Math.max(0, boostedVal - 0.08);
      const compressed = Math.log10(1 + cleaned * 10) / Math.log10(101);
      return Math.min(255, Math.max(0, compressed * 255));
    });
  }

  public calculateRobustThreshold(scores: number[]) {
    if (scores.length < 10) return 3.5;
    const sorted = [...scores].sort((a, b) => a - b);
    const baseline = sorted[Math.floor(sorted.length * 0.65)];
    const absoluteDeviations = scores.map(s => Math.abs(s - baseline));
    const sortedDeviations = [...absoluteDeviations].sort((a, b) => a - b);
    const mad = sortedDeviations[Math.floor(sortedDeviations.length / 2)];
    const multiplier = 6.0 - (this.sensitivity * 0.4);
    const sigmaEst = mad * 1.4826;
    return Math.max(0.5, baseline + (multiplier * sigmaEst));
  }

  private updateThreshold() {
    if (this.errorStats.length < 20) return;
    this.threshold = this.calculateRobustThreshold(this.errorStats);
  }

  public async trainOnFile(frames: number[][], label: 'Normal' | 'Anomaly') {
    if (!this.model) return { error: 0 };
    const sequences: number[][][] = [];
    for (let i = 0; i <= frames.length - this.windowSize; i += 4) {
      sequences.push(frames.slice(i, i + this.windowSize));
    }
    if (sequences.length === 0) return { error: 0 };

    return tf.tidy(() => {
      const xs = tf.tensor3d(sequences);
      const normXs = xs.div(tf.scalar(255));
      
      if (label === 'Normal') {
        this.model!.fit(normXs, normXs, { epochs: 1, batchSize: 8, verbose: 0 }).then(() => {
           this.totalProcessedFiles++;
           this.persist();
        });
      }

      const predictions = this.model!.predict(normXs) as tf.Tensor;
      const squaredDiff = tf.square(tf.sub(normXs, predictions));
      const errors = squaredDiff.mean([1, 2]); 
      // Fix: Cast errorData to number[] to avoid 'unknown' type in reduce during arithmetic operations
      const errorData = Array.from(errors.dataSync()) as number[];
      const avgError = (errorData.reduce((a: number, b: number) => a + b, 0) / errorData.length) * 1000;

      if (label === 'Normal') {
        for(let i = 0; i < errorData.length; i += 2) {
          this.errorStats.push(errorData[i] * 1000);
        }
        if (this.errorStats.length > 1000) this.errorStats = this.errorStats.slice(-1000);
        this.updateThreshold();
      }
      return { error: avgError };
    });
  }

  public async predict(sequence: number[][]) {
    if (!this.model) return { score: 0 };
    return tf.tidy(() => {
      const input = tf.tensor3d([sequence]);
      const normInput = input.div(tf.scalar(255));
      const output = this.model!.predict(normInput) as tf.Tensor;
      const score = tf.losses.meanSquaredError(normInput, output).dataSync()[0] * 1000;
      return { score };
    });
  }

  private async persist() {
    if (this.model) {
      try {
        await this.model.save(STORAGE_PATH);
        localStorage.setItem('sentinel_stats', JSON.stringify({
          errorStats: this.errorStats,
          totalProcessedFiles: this.totalProcessedFiles,
          threshold: this.threshold,
          sensitivity: this.sensitivity
        }));
      } catch (e) {}
    }
  }

  public async importModelFromFiles(jsonFile: File, weightsFile: File, statsFile?: File) {
    try {
      const loadedModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
      loadedModel.compile({
        optimizer: tf.train.adam(this.config.learningRate),
        loss: 'meanSquaredError'
      });
      this.model = loadedModel;
      this.isModelLoaded = true;

      if (statsFile) {
        const text = await statsFile.text();
        const parsed = JSON.parse(text);
        this.errorStats = parsed.errorStats || [];
        this.totalProcessedFiles = parsed.totalProcessedFiles || 0;
        this.threshold = parsed.threshold || 3.5;
        this.sensitivity = parsed.sensitivity || 5.0;
        localStorage.setItem('sentinel_stats', text);
      }
      
      await this.persist();
      return true;
    } catch (e) {
      console.error("Import error:", e);
      return false;
    }
  }

  public async clearKnowledge() {
    this.errorStats = [];
    this.totalProcessedFiles = 0;
    this.threshold = 3.5;
    localStorage.removeItem('sentinel_stats');
    try { await tf.io.removeModel(STORAGE_PATH); } catch {}
    await this.createNewModel();
  }

  public async exportModel() {
    if (!this.model) return;
    // Eksport modelu (pobierze model.json i model.weights.bin)
    await this.model.save('downloads://sentinel-model');
    
    // Eksport statystyk do pliku JSON
    const stats = {
      errorStats: this.errorStats,
      totalProcessedFiles: this.totalProcessedFiles,
      threshold: this.threshold,
      sensitivity: this.sensitivity
    };
    const blob = new Blob([JSON.stringify(stats, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sentinel_stats.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  public getMemInfo() { 
    try { return tf.memory(); } catch { return { numBytes: 0 }; }
  }
}

export const detector = new AnomalyDetector();
