import { Response } from 'express';
import { ApiResponse, ResponseMeta, ValidationError } from '../types';

export const sendSuccess = <T>(
  res: Response,
  data: T,
  message = 'Success',
  statusCode = 200,
  meta?: ResponseMeta
): Response => {
  const response: ApiResponse<T> = {
    success: true,
    message,
    data,
    ...(meta && { meta }),
  };
  return res.status(statusCode).json(response);
};

export const sendCreated = <T>(
  res: Response,
  data: T,
  message = 'Created successfully'
): Response => {
  return sendSuccess(res, data, message, 201);
};

export const sendError = (
  res: Response,
  message: string,
  statusCode = 500,
  error?: string
): Response => {
  const response: ApiResponse = {
    success: false,
    message,
    ...(error && { error }),
  };
  return res.status(statusCode).json(response);
};

export const sendValidationError = (
  res: Response,
  errors: ValidationError[],
  message = 'Validation failed'
): Response => {
  const response: ApiResponse = {
    success: false,
    message,
    errors,
  };
  return res.status(422).json(response);
};

export const sendUnauthorized = (
  res: Response,
  message = 'Unauthorized'
): Response => {
  return sendError(res, message, 401);
};

export const sendForbidden = (
  res: Response,
  message = 'Forbidden'
): Response => {
  return sendError(res, message, 403);
};

export const sendNotFound = (
  res: Response,
  message = 'Resource not found'
): Response => {
  return sendError(res, message, 404);
};
