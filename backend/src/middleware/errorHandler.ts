import { Request, Response, NextFunction } from 'express';
import { logger } from '../utils/logger';
import { env } from '../config/env';

export interface AppError extends Error {
  statusCode?: number;
  isOperational?: boolean;
}

export const createError = (message: string, statusCode: number): AppError => {
  const error: AppError = new Error(message);
  error.statusCode = statusCode;
  error.isOperational = true;
  return error;
};

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export const errorHandler = (
  err: AppError,
  _req: Request,
  res: Response,
  _next: NextFunction
): void => {
  const statusCode = err.statusCode ?? 500;
  const message = err.message || 'Internal Server Error';

  logger.error({
    message: err.message,
    statusCode,
    stack: env.isDevelopment ? err.stack : undefined,
  });

  // Mongoose validation error
  if (err.name === 'ValidationError') {
    res.status(422).json({
      success: false,
      message: 'Validation error',
      error: err.message,
    });
    return;
  }

  // Mongoose duplicate key error
  if ((err as NodeJS.ErrnoException).code === '11000') {
    res.status(409).json({
      success: false,
      message: 'A resource with that value already exists',
    });
    return;
  }

  // JWT errors
  if (err.name === 'JsonWebTokenError' || err.name === 'TokenExpiredError') {
    res.status(401).json({
      success: false,
      message: 'Invalid or expired token',
    });
    return;
  }

  res.status(statusCode).json({
    success: false,
    message: err.isOperational ? message : 'An unexpected error occurred',
    ...(env.isDevelopment && { stack: err.stack }),
  });
};

export const notFoundHandler = (_req: Request, res: Response): void => {
  res.status(404).json({
    success: false,
    message: 'Route not found',
  });
};
