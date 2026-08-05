import axios, { AxiosError } from 'axios';
import { env } from '../config/env';
import { logger } from '../utils/logger';
import { AIPredictRequest, AIPredictResponse, AIExplainRequest, AIExplainResponse } from '../types';

/**
 * AI Service — HTTP proxy layer to the Python FastAPI microservice.
 *
 * These methods will communicate with the EQ-KA-GCN model once
 * the AI microservice is fully implemented. For now, they return
 * realistic placeholder responses to enable frontend development.
 *
 * To integrate the real AI model:
 * 1. Set AI_SERVICE_URL in .env to the running FastAPI instance
 * 2. Remove the placeholder response below
 * 3. Uncomment the actual axios call
 */

const aiClient = axios.create({
  baseURL: env.AI_SERVICE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

aiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    logger.error(`AI Service error: ${error.message}`, {
      url: error.config?.url,
      status: error.response?.status,
    });
    throw error;
  }
);

export const predictToxicity = async (
  request: AIPredictRequest
): Promise<AIPredictResponse> => {
  try {
    logger.info(`[AI Proxy] Requesting prediction from AI service for SMILES: ${request.smiles.substring(0, 30)}`);
    const { data } = await aiClient.post<AIPredictResponse>('/api/predict', request);
    return data;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'AI microservice unavailable';
    logger.error(`Prediction proxy failed: ${message}`);
    throw new Error(`AI prediction service error: ${message}`);
  }
};

export const explainPrediction = async (
  request: AIExplainRequest
): Promise<AIExplainResponse> => {
  try {
    logger.info(`[AI Proxy] Requesting explanation from AI service for SMILES: ${request.smiles.substring(0, 30)}`);
    const { data } = await aiClient.post<AIExplainResponse>('/api/explain', request);
    return data;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'AI microservice unavailable';
    logger.error(`Explanation proxy failed: ${message}`);
    throw new Error(`AI explanation service error: ${message}`);
  }
};

export const getAIServiceStatus = async (): Promise<{
  status: 'online' | 'offline';
  version?: string;
}> => {
  try {
    const { data } = await aiClient.get('/health', { timeout: 3000 });
    return { status: 'online', version: data?.version };
  } catch {
    return { status: 'offline' };
  }
};
