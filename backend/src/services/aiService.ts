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
  const startTime = Date.now();

  try {
    // =====================================================
    // PLACEHOLDER — Replace with real call when AI service
    // is running:
    //
    // const { data } = await aiClient.post<AIPredictResponse>('/api/predict', request);
    // return data;
    // =====================================================

    logger.info(`[AI Placeholder] Running mock prediction for SMILES: ${request.smiles.substring(0, 30)}...`);

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 500 + Math.random() * 1000));

    // Deterministic mock based on SMILES length for consistency
    const hash = request.smiles.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const isToxic = hash % 2 === 0;
    const probability = isToxic ? 0.65 + (hash % 30) / 100 : 0.15 + (hash % 25) / 100;
    const atomCount = Math.min(request.smiles.replace(/[^A-Z]/g, '').length, 20);

    return {
      prediction: isToxic ? 'toxic' : 'non-toxic',
      probability: parseFloat(probability.toFixed(4)),
      confidence: parseFloat((0.75 + (hash % 20) / 100).toFixed(4)),
      importantAtoms: Array.from({ length: Math.min(atomCount, 5) }, (_, i) => i * 2),
      importantBonds: Array.from({ length: Math.min(atomCount - 1, 4) }, (_, i) => ({
        atomA: i,
        atomB: i + 1,
        weight: parseFloat((0.4 + (i * 0.1)).toFixed(3)),
      })),
      executionTime: Date.now() - startTime,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'AI service unavailable';
    logger.error(`Prediction failed: ${message}`);
    throw new Error(`AI prediction service error: ${message}`);
  }
};

export const explainPrediction = async (
  request: AIExplainRequest
): Promise<AIExplainResponse> => {
  try {
    // =====================================================
    // PLACEHOLDER — Replace with real call when AI service
    // is running:
    //
    // const { data } = await aiClient.post<AIExplainResponse>('/api/explain', request);
    // return data;
    // =====================================================

    logger.info(`[AI Placeholder] Running mock explanation for SMILES: ${request.smiles.substring(0, 30)}...`);

    await new Promise((resolve) => setTimeout(resolve, 300 + Math.random() * 500));

    const atomCount = Math.min(request.smiles.replace(/[^A-Z]/g, '').length, 15);

    return {
      atomAttentions: Array.from({ length: atomCount }, (_, i) => ({
        atomIndex: i,
        weight: parseFloat((Math.random() * 0.9 + 0.1).toFixed(4)),
      })),
      bondAttentions: Array.from({ length: Math.max(atomCount - 1, 0) }, (_, i) => ({
        bondIndex: i,
        weight: parseFloat((Math.random() * 0.8 + 0.1).toFixed(4)),
        atoms: [i, i + 1] as [number, number],
      })),
      saliencyMap: [],
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'AI service unavailable';
    logger.error(`Explanation failed: ${message}`);
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
