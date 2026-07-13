import { Request, Response, NextFunction } from 'express';
import { validationResult } from 'express-validator';
import { User } from '../models/User';
import { Prediction } from '../models/Prediction';
import { AuthRequest, ValidationError } from '../types';
import {
  sendSuccess,
  sendError,
  sendNotFound,
  sendValidationError,
} from '../utils/response';
import { logger } from '../utils/logger';

const formatValidationErrors = (result: ReturnType<typeof validationResult>): ValidationError[] =>
  result.array().map((err) => ({
    field: err.type === 'field' ? err.path : 'unknown',
    message: err.msg,
  }));

export const getProfile = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const userId = (req as AuthRequest).user?.userId;
    const user = await User.findById(userId);

    if (!user) {
      sendNotFound(res, 'User not found');
      return;
    }

    const predictionCount = await Prediction.countDocuments({ user: userId });

    sendSuccess(res, { user, predictionCount }, 'Profile retrieved successfully');
  } catch (error) {
    next(error);
  }
};

export const updateProfile = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      sendValidationError(res, formatValidationErrors(errors));
      return;
    }

    const userId = (req as AuthRequest).user?.userId;
    const { name, email } = req.body;

    if (email) {
      const existingUser = await User.findOne({ email, _id: { $ne: userId } });
      if (existingUser) {
        sendError(res, 'Email is already in use by another account', 409);
        return;
      }
    }

    const user = await User.findByIdAndUpdate(
      userId,
      { ...(name && { name }), ...(email && { email }) },
      { new: true, runValidators: true }
    );

    if (!user) {
      sendNotFound(res, 'User not found');
      return;
    }

    logger.info(`Profile updated for user: ${user.email}`);
    sendSuccess(res, { user }, 'Profile updated successfully');
  } catch (error) {
    next(error);
  }
};

export const changePassword = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      sendValidationError(res, formatValidationErrors(errors));
      return;
    }

    const userId = (req as AuthRequest).user?.userId;
    const { currentPassword, newPassword } = req.body;

    const user = await User.findById(userId).select('+password');
    if (!user) {
      sendNotFound(res, 'User not found');
      return;
    }

    const isValid = await user.comparePassword(currentPassword);
    if (!isValid) {
      sendError(res, 'Current password is incorrect', 400);
      return;
    }

    user.password = newPassword;
    await user.save();

    // Invalidate all refresh tokens on password change
    await User.findByIdAndUpdate(userId, { refreshTokens: [] });
    res.clearCookie('refreshToken', { path: '/api/auth' });

    logger.info(`Password changed for user: ${user.email}`);
    sendSuccess(res, null, 'Password changed successfully. Please log in again.');
  } catch (error) {
    next(error);
  }
};

export const deleteAccount = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const userId = (req as AuthRequest).user?.userId;

    const user = await User.findByIdAndDelete(userId);
    if (!user) {
      sendNotFound(res, 'User not found');
      return;
    }

    // Cascade delete predictions
    await Prediction.deleteMany({ user: userId });

    res.clearCookie('refreshToken', { path: '/api/auth' });
    logger.info(`Account deleted for user: ${user.email}`);
    sendSuccess(res, null, 'Account deleted successfully');
  } catch (error) {
    next(error);
  }
};
