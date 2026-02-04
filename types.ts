
export interface Anomaly {
  id: string;
  timestamp: Date;
  offsetSeconds?: number;
  durationSeconds: number;
  intensity: number;
  severity: 'High' | 'Medium' | 'Low';
  description: string;
  type: string;
  spectralData?: number[][];
  audioUrl?: string;
}

export interface AudioChartData {
  time: string;
  amplitude: number;
  anomalyLevel: number;
  second?: number;
}

export interface ModelConfig {
  epochs: number;
  learningRate: number;
  batchSize: number;
  latentDim: number;
}
