import api from './api';
import {
  ApiResponse,
  Prediction,
  PredictionStats,
  PaginationParams,
  ResponseMeta,
  ModelStatus,
} from '@/types';

export const predictionService = {
  async createPrediction(smiles: string): Promise<Prediction> {
    const { data } = await api.post<ApiResponse<{ prediction: Prediction }>>('/predictions', { smiles });
    return data.data!.prediction;
  },

  async getPredictions(params?: PaginationParams): Promise<{
    predictions: Prediction[];
    meta: ResponseMeta;
  }> {
    const { data } = await api.get<ApiResponse<{ predictions: Prediction[] }>>('/predictions', {
      params,
    });
    return {
      predictions: data.data!.predictions,
      meta: data.meta!,
    };
  },

  async getPredictionById(id: string): Promise<Prediction> {
    const { data } = await api.get<ApiResponse<{ prediction: Prediction }>>(`/predictions/${id}`);
    return data.data!.prediction;
  },

  async deletePrediction(id: string): Promise<void> {
    await api.delete(`/predictions/${id}`);
  },

  async getStats(): Promise<PredictionStats> {
    const { data } = await api.get<ApiResponse<{ stats: PredictionStats }>>('/predictions/stats');
    return data.data!.stats;
  },

  async getModelStatus(): Promise<ModelStatus> {
    const { data } = await api.get<ApiResponse<ModelStatus>>('/ai/status');
    return data.data!;
  },
};
