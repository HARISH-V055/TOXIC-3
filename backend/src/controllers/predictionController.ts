import { Request, Response, NextFunction } from 'express';
import { validationResult } from 'express-validator';
import mongoose from 'mongoose';
import { Prediction } from '../models/Prediction';
import { predictToxicity } from '../services/aiService';
import { AuthRequest, ValidationError, PaginationQuery } from '../types';
import {
  sendSuccess,
  sendCreated,
  sendError,
  sendNotFound,
  sendForbidden,
  sendValidationError,
} from '../utils/response';
import { logger } from '../utils/logger';

const formatValidationErrors = (result: ReturnType<typeof validationResult>): ValidationError[] =>
  result.array().map((err) => ({
    field: err.type === 'field' ? err.path : 'unknown',
    message: err.msg,
  }));

export const createPrediction = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      sendValidationError(res, formatValidationErrors(errors));
      return;
    }

    const userId = (req as AuthRequest).user?.userId;
    const { smiles } = req.body;

    // Create pending record
    const prediction = await Prediction.create({
      user: userId,
      smiles,
      prediction: 'pending',
    });

    logger.info(`Prediction submitted by user ${userId} for SMILES: ${smiles.substring(0, 30)}`);

    // Call AI service
    try {
      const aiResult = await predictToxicity({ smiles });

      const updated = await Prediction.findByIdAndUpdate(
        prediction._id,
        {
          prediction: aiResult.prediction,
          probability: aiResult.probability,
          confidence: aiResult.confidence,
          importantAtoms: aiResult.importantAtoms,
          importantBonds: aiResult.importantBonds,
          executionTime: aiResult.executionTime,
        },
        { new: true }
      );

      sendCreated(res, { prediction: updated }, 'Prediction completed successfully');
    } catch (aiError) {
      // Mark as error but don't lose the record
      await Prediction.findByIdAndUpdate(prediction._id, { prediction: 'error' });
      const message = aiError instanceof Error ? aiError.message : 'AI service failed';
      sendError(res, `Prediction failed: ${message}`, 503);
    }
  } catch (error) {
    next(error);
  }
};

export const getPredictions = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const userId = (req as AuthRequest).user?.userId;
    const {
      page = '1',
      limit = '10',
      search = '',
      sort = 'createdAt',
      order = 'desc',
    } = req.query as PaginationQuery;

    const pageNum = Math.max(parseInt(page, 10), 1);
    const limitNum = Math.min(parseInt(limit, 10), 50);
    const skip = (pageNum - 1) * limitNum;

    const filterQuery: mongoose.FilterQuery<typeof Prediction> = { user: userId };
    if (search) {
      filterQuery.smiles = { $regex: search, $options: 'i' };
    }

    const sortOrder = order === 'asc' ? 1 : -1;
    const allowedSortFields = ['createdAt', 'prediction', 'confidence', 'probability'];
    const sortField = allowedSortFields.includes(sort) ? sort : 'createdAt';

    const [predictions, total] = await Promise.all([
      Prediction.find(filterQuery)
        .sort({ [sortField]: sortOrder })
        .skip(skip)
        .limit(limitNum)
        .lean(),
      Prediction.countDocuments(filterQuery),
    ]);

    sendSuccess(
      res,
      { predictions },
      'Predictions retrieved successfully',
      200,
      {
        total,
        page: pageNum,
        limit: limitNum,
        pages: Math.ceil(total / limitNum),
      }
    );
  } catch (error) {
    next(error);
  }
};

export const getPredictionById = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const userId = (req as AuthRequest).user?.userId;
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      sendError(res, 'Invalid prediction ID', 400);
      return;
    }

    const prediction = await Prediction.findById(id);
    if (!prediction) {
      sendNotFound(res, 'Prediction not found');
      return;
    }

    if (prediction.user.toString() !== userId) {
      sendForbidden(res, 'Access denied');
      return;
    }

    sendSuccess(res, { prediction }, 'Prediction retrieved successfully');
  } catch (error) {
    next(error);
  }
};

export const deletePrediction = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const userId = (req as AuthRequest).user?.userId;
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      sendError(res, 'Invalid prediction ID', 400);
      return;
    }

    const prediction = await Prediction.findById(id);
    if (!prediction) {
      sendNotFound(res, 'Prediction not found');
      return;
    }

    if (prediction.user.toString() !== userId) {
      sendForbidden(res, 'Access denied');
      return;
    }

    await prediction.deleteOne();
    logger.info(`Prediction ${id} deleted by user ${userId}`);
    sendSuccess(res, null, 'Prediction deleted successfully');
  } catch (error) {
    next(error);
  }
};

export const getPredictionStats = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const userId = (req as AuthRequest).user?.userId;

    const stats = await Prediction.aggregate([
      { $match: { user: new mongoose.Types.ObjectId(userId) } },
      {
        $group: {
          _id: null,
          total: { $sum: 1 },
          toxic: { $sum: { $cond: [{ $eq: ['$prediction', 'toxic'] }, 1, 0] } },
          nonToxic: { $sum: { $cond: [{ $eq: ['$prediction', 'non-toxic'] }, 1, 0] } },
          pending: { $sum: { $cond: [{ $eq: ['$prediction', 'pending'] }, 1, 0] } },
          avgConfidence: { $avg: '$confidence' },
          avgExecutionTime: { $avg: '$executionTime' },
        },
      },
    ]);

    const result = stats[0] ?? {
      total: 0,
      toxic: 0,
      nonToxic: 0,
      pending: 0,
      avgConfidence: 0,
      avgExecutionTime: 0,
    };

    sendSuccess(res, { stats: result }, 'Statistics retrieved successfully');
  } catch (error) {
    next(error);
  }
};
