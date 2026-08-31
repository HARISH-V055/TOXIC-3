import { Request, Response, NextFunction } from 'express';
import { validationResult } from 'express-validator';
import { predictToxicity, explainPrediction, getAIServiceStatus } from '../services/aiService';
import { ModelInformation } from '../models/ModelInformation';
import { sendSuccess, sendValidationError } from '../utils/response';
import { ValidationError } from '../types';
import { logger } from '../utils/logger';

const formatValidationErrors = (result: ReturnType<typeof validationResult>): ValidationError[] =>
  result.array().map((err) => ({
    field: err.type === 'field' ? err.path : 'unknown',
    message: err.msg,
  }));

export const predict = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      sendValidationError(res, formatValidationErrors(errors));
      return;
    }

    const { smiles } = req.body;

    logger.info(`[AI API] Direct predict request for SMILES: ${smiles.substring(0, 30)}`);

    const result = await predictToxicity({ smiles });
    sendSuccess(res, result, 'Prediction completed');
  } catch (error) {
    next(error);
  }
};

export const explain = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      sendValidationError(res, formatValidationErrors(errors));
      return;
    }

    const { smiles, predictionId } = req.body;

    logger.info(`[AI API] Explain request for SMILES: ${smiles.substring(0, 30)}`);

    const result = await explainPrediction({ smiles, predictionId });
    sendSuccess(res, result, 'Explanation generated');
  } catch (error) {
    next(error);
  }
};

export const getModelStatus = async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const [serviceStatus, modelInfo] = await Promise.all([
      getAIServiceStatus(),
      ModelInformation.getActive(),
    ]);

    const isServiceOnline = serviceStatus.status === 'online';
    const isModelLoaded = Boolean(serviceStatus.modelLoaded);

    sendSuccess(
      res,
      {
        service: serviceStatus,
        model: modelInfo ?? {
          version: serviceStatus.version || '0.1.0-quantized',
          name: 'EQ-KA-GCN',
          status: (isServiceOnline && isModelLoaded) ? 'active' : 'offline',
          description: (isServiceOnline && isModelLoaded)
            ? 'EQ-KA-GCN with Fourier-KAN & Quantization-Aware Training active.'
            : 'AI model integration pending. Architecture ready for deployment.',
        },
      },
      'Model status retrieved'
    );
  } catch (error) {
    next(error);
  }
};
