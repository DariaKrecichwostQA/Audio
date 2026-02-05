
import * as tf from '@tensorflow/tfjs';
import { ModelConfig } from '../types';

const STORAGE_PATH = 'indexeddb://sentinel-model-v1';

class AnomalyDetector {
  private model: tf.LayersModel | null = null;
  private config: ModelConfig = {
    epochs: 5,
    learningRate: 0.001,
    batchSize: 32,
    latentDim: 24
  };
  private threshold: number = 3.5; 
  private sensitivity: number = 5.0; 
  private windowSize: number = 12;
  private errorStats: number[] = [];
  private totalProcessedFiles: number = 0;
  private isModelLoaded: boolean = false;

  constructor() {
    this.init();
  }

  private async init() {
    try {
      await tf.ready();
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
      
      const savedStats = localStorage.getItem('sentinel_stats');
      if (savedStats) {
        const parsed = JSON.parse(savedStats);
        this.errorStats = parsed.errorStats || [];
        this.totalProcessedFiles = parsed.totalProcessedFiles || 0;
        this.threshold = parsed.threshold || 3.5;
        this.sensitivity = parsed.sensitivity || 5.0;
      }
    } catch (e) {
      console.error("Inicjalizacja błąd:", e);
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
  public getBackendName() { return tf.getBackend().toUpperCase(); }
  public getIsLoadedFromStorage() { return this.isModelLoaded; }

  public scaleFrame(frame: number[]) {
    return frame.map(v => {
      const val = Math.max(0, v - 0.10); // Lżejsza redukcja szumu tła
      const compressed = Math.log10(1 + val * 5) / Math.log10(50); 
      return Math.min(255, Math.max(0, compressed * 255));
    });
  }

  public calculateRobustThreshold(scores: number[]) {
    if (scores.length < 10) return 3.5;
    
    const sorted = [...scores].sort((a, b) => a - b);
    
    // ZMIANA: Przesunięcie na 65. percentyl (bliżej sygnału niż Q3)
    const baselineIndex = Math.floor(sorted.length * 0.65);
    const baseline = sorted[baselineIndex];
    
    const absoluteDeviations = scores.map(s => Math.abs(s - baseline));
    const sortedDeviations = [...absoluteDeviations].sort((a, b) => a - b);
    const mad = sortedDeviations[Math.floor(sortedDeviations.length / 2)];
    
    // ZMIANA: Znacznie mniejszy mnożnik (wcześniej dochodził do 10-14, teraz max 6)
    const multiplier = 6.0 - (this.sensitivity * 0.4);
    const sigmaEst = mad * 1.4826;
    
    // Safety floor obniżony na 0.5
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

    const xs = tf.tensor3d(sequences);
    const normXs = xs.div(tf.scalar(255));

    try {
      if (label === 'Normal') {
        await this.model.fit(normXs, normXs, { 
          epochs: 5, 
          batchSize: 32, 
          verbose: 0,
          shuffle: true 
        });
        this.totalProcessedFiles++;
      }

      const predictions = this.model.predict(normXs) as tf.Tensor;
      const errors = tf.losses.meanSquaredError(normXs, predictions).mean(1);
      const errorData = await errors.data();
      const typedErrorData = errorData as Float32Array;
      const avgError = (Array.from(typedErrorData).reduce((a: number, b: number) => a + b, 0) / typedErrorData.length) * 1000;
      
      if (label === 'Normal') {
        for(let i = 0; i < typedErrorData.length; i += 2) {
          this.errorStats.push((typedErrorData[i] as number) * 1000);
        }
        if (this.errorStats.length > 5000) this.errorStats = this.errorStats.slice(-5000);
        this.updateThreshold();
      }

      predictions.dispose();
      errors.dispose();
      
      this.persist();
      return { error: avgError };
    } finally {
      xs.dispose();
      normXs.dispose();
    }
  }

  public async predict(sequence: number[][]) {
    if (!this.model) return { score: 0 };
    return tf.tidy(() => {
      const input = tf.tensor3d([sequence]);
      const normInput = input.div(tf.scalar(255));
      const output = this.model!.predict(normInput) as tf.Tensor;
      const error = tf.losses.meanSquaredError(normInput, output);
      const data = error.dataSync() as Float32Array;
      return { score: (data[0] as number) * 1000 };
    });
  }

  private async persist() {
    if (this.model) {
      await this.model.save(STORAGE_PATH);
      localStorage.setItem('sentinel_stats', JSON.stringify({
        errorStats: this.errorStats,
        totalProcessedFiles: this.totalProcessedFiles,
        threshold: this.threshold,
        sensitivity: this.sensitivity
      }));
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
    if (this.model) await this.model.save('downloads://sentinel-model');
  }

  public getMemInfo() {
    return tf.memory();
  }
}

export const detector = new AnomalyDetector();
