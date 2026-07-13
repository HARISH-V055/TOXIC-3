import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { Prediction, PredictionStats, ResponseMeta, ModelStatus } from '@/types';

interface PredictionState {
  predictions: Prediction[];
  currentPrediction: Prediction | null;
  stats: PredictionStats | null;
  modelStatus: ModelStatus | null;
  meta: ResponseMeta | null;
  isLoading: boolean;
  isPredicting: boolean;
  error: string | null;

  // Actions
  setPredictions: (predictions: Prediction[], meta: ResponseMeta) => void;
  addPrediction: (prediction: Prediction) => void;
  removePrediction: (id: string) => void;
  setCurrentPrediction: (prediction: Prediction | null) => void;
  setStats: (stats: PredictionStats) => void;
  setModelStatus: (status: ModelStatus) => void;
  setLoading: (loading: boolean) => void;
  setPredicting: (predicting: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialState = {
  predictions: [],
  currentPrediction: null,
  stats: null,
  modelStatus: null,
  meta: null,
  isLoading: false,
  isPredicting: false,
  error: null,
};

export const usePredictionStore = create<PredictionState>()(
  devtools(
    (set) => ({
      ...initialState,

      setPredictions: (predictions, meta) =>
        set({ predictions, meta }, false, 'setPredictions'),

      addPrediction: (prediction) =>
        set(
          (state) => ({
            predictions: [prediction, ...state.predictions],
            currentPrediction: prediction,
          }),
          false,
          'addPrediction'
        ),

      removePrediction: (id) =>
        set(
          (state) => ({
            predictions: state.predictions.filter((p) => p._id !== id),
          }),
          false,
          'removePrediction'
        ),

      setCurrentPrediction: (prediction) =>
        set({ currentPrediction: prediction }, false, 'setCurrentPrediction'),

      setStats: (stats) =>
        set({ stats }, false, 'setStats'),

      setModelStatus: (modelStatus) =>
        set({ modelStatus }, false, 'setModelStatus'),

      setLoading: (isLoading) =>
        set({ isLoading }, false, 'setLoading'),

      setPredicting: (isPredicting) =>
        set({ isPredicting }, false, 'setPredicting'),

      setError: (error) =>
        set({ error }, false, 'setError'),

      reset: () =>
        set(initialState, false, 'reset'),
    }),
    { name: 'PredictionStore' }
  )
);
