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
    const { data } = await aiClient.post<any>('/api/predict', request);

    const importantAtoms = (data.importantAtoms ?? data.important_atoms ?? []).map((a: any) => ({
      index: typeof a === 'number' ? a : (a.index ?? a.atom_index ?? 0),
      element: typeof a === 'number' ? 'C' : (a.element ?? a.atom_symbol ?? 'C'),
      score: typeof a === 'number' ? 1.0 : (typeof a.score === 'number' ? a.score : (a.importance_score ?? 1.0)),
    }));

    const importantBonds = (data.importantBonds ?? data.important_bonds ?? []).map((b: any) => ({
      source: b.source ?? b.atomA ?? b.u ?? 0,
      target: b.target ?? b.atomB ?? b.v ?? 0,
      score: typeof b.score === 'number' ? b.score : (b.weight ?? b.importance_score ?? 1.0),
    }));

    const molecularGraph = data.molecularGraph ?? data.molecular_graph ?? { atoms: [], bonds: [] };
    const inferenceTimeMs = data.inferenceTimeMs ?? data.inference_time_ms ?? data.executionTime ?? data.execution_time ?? 0;

    return {
      smiles: data.smiles ?? request.smiles,
      prediction: data.prediction,
      probability: data.probability,
      confidence: data.confidence,
      threshold: data.threshold ?? 0.75,
      endpoint: data.endpoint ?? 'Tox21 SR-p53',
      inferenceTimeMs,
      importantAtoms,
      importantBonds,
      molecularGraph,
      explanationImage: data.explanationImage ?? data.explanation_image ?? '/outputs/explanations/molecule_explanation.png',
    };
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
    const { data } = await aiClient.post<any>('/api/explain', request);

    const atomAttentions = (data.atomAttentions ?? data.atom_attentions ?? []).map((a: any) => ({
      atomIndex: a.atomIndex ?? a.atom_index,
      weight: a.weight,
    }));
    const bondAttentions = (data.bondAttentions ?? data.bond_attentions ?? []).map((b: any) => ({
      bondIndex: b.bondIndex ?? b.bond_index,
      weight: b.weight,
      atoms: b.atoms,
    }));
    const saliencyMap = data.saliencyMap ?? data.saliency_map ?? [];

    return {
      atomAttentions,
      bondAttentions,
      saliencyMap,
    };
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
