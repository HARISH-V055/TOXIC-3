import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { env } from '../config/env';
import { AuthRequest, JwtPayload } from '../types';
import { sendUnauthorized } from '../utils/response';
import { logger } from '../utils/logger';

export const authenticate = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      sendUnauthorized(res, 'No token provided');
      return;
    }

    const token = authHeader.split(' ')[1];

    const decoded = jwt.verify(token, env.JWT_ACCESS_SECRET) as JwtPayload;
    (req as AuthRequest).user = decoded;
    next();
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      sendUnauthorized(res, 'Token has expired');
      return;
    }
    if (error instanceof jwt.JsonWebTokenError) {
      sendUnauthorized(res, 'Invalid token');
      return;
    }
    logger.error('Authentication middleware error:', error);
    sendUnauthorized(res, 'Authentication failed');
  }
};

export const authorizeRoles = (...roles: string[]) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const user = (req as AuthRequest).user;
    if (!user || !roles.includes(user.role)) {
      res.status(403).json({
        success: false,
        message: 'You do not have permission to perform this action',
      });
      return;
    }
    next();
  };
};
