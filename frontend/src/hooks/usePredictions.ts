import { useCallback } from 'react';
import { usePredictionStore } from '@store/usePredictionStore';
import { predictionService } from '@services/predictionService';
import { PaginationParams } from '@/types';

export const usePredictions = () => {
  const {
    predictions,
    currentPrediction,
    stats,
    modelStatus,
    meta,
    isLoading,
    isPredicting,
    error,
    setPredictions,
    addPrediction,
    removePrediction,
    setCurrentPrediction,
    setStats,
    setModelStatus,
    setLoading,
    setPredicting,
    setError,
  } = usePredictionStore();

  const fetchPredictions = useCallback(
    async (params?: PaginationParams) => {
      setLoading(true);
      setError(null);
      try {
        const { predictions, meta } = await predictionService.getPredictions(params);
        setPredictions(predictions, meta);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch predictions';
        setError(message);
      } finally {
        setLoading(false);
      }
    },
    [setPredictions, setLoading, setError]
  );

  const predict = useCallback(
    async (smiles: string) => {
      setPredicting(true);
      setError(null);
      try {
        const prediction = await predictionService.createPrediction(smiles);
        addPrediction(prediction);
        return prediction;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Prediction failed';
        setError(message);
        throw err;
      } finally {
        setPredicting(false);
      }
    },
    [addPrediction, setPredicting, setError]
  );

  const deletePrediction = useCallback(
    async (id: string) => {
      try {
        await predictionService.deletePrediction(id);
        removePrediction(id);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to delete prediction';
        setError(message);
        throw err;
      }
    },
    [removePrediction, setError]
  );

  const fetchStats = useCallback(async () => {
    try {
      const stats = await predictionService.getStats();
      setStats(stats);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  }, [setStats]);

  const fetchModelStatus = useCallback(async () => {
    try {
      const status = await predictionService.getModelStatus();
      setModelStatus(status);
    } catch (err) {
      console.error('Failed to fetch model status:', err);
    }
  }, [setModelStatus]);

  return {
    predictions,
    currentPrediction,
    stats,
    modelStatus,
    meta,
    isLoading,
    isPredicting,
    error,
    setCurrentPrediction,
    fetchPredictions,
    predict,
    deletePrediction,
    fetchStats,
    fetchModelStatus,
  };
};
