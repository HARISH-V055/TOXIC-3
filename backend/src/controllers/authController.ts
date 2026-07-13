import { Request, Response, NextFunction } from 'express';
import { validationResult } from 'express-validator';
import { User } from '../models/User';
import {
  generateTokenPair,
  verifyRefreshToken,
  getRefreshTokenCookieOptions,
} from '../services/tokenService';
import {
  sendSuccess,
  sendCreated,
  sendError,
  sendValidationError,
  sendUnauthorized,
} from '../utils/response';
import { logger } from '../utils/logger';
import { UserRole, ValidationError } from '../types';

const formatValidationErrors = (result: ReturnType<typeof validationResult>): ValidationError[] =>
  result.array().map((err) => ({
    field: err.type === 'field' ? err.path : 'unknown',
    message: err.msg,
  }));

export const register = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      sendValidationError(res, formatValidationErrors(errors));
      return;
    }

    const { name, email, password } = req.body;

    const existingUser = await User.findOne({ email: email.toLowerCase() });
    if (existingUser) {
      sendError(res, 'A user with this email already exists', 409);
      return;
    }

    const user = await User.create({ name, email, password, role: UserRole.USER });
    const { accessToken, refreshToken } = generateTokenPair(
      user._id.toString(),
      user.email,
      user.role
    );

    // Store refresh token
    await User.findByIdAndUpdate(user._id, {
      $push: { refreshTokens: refreshToken },
    });

    res.cookie('refreshToken', refreshToken, getRefreshTokenCookieOptions());

    logger.info(`New user registered: ${user.email}`);

    sendCreated(
      res,
      {
        user: { id: user._id, name: user.name, email: user.email, role: user.role },
        accessToken,
      },
      'Account created successfully'
    );
  } catch (error) {
    next(error);
  }
};

export const login = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      sendValidationError(res, formatValidationErrors(errors));
      return;
    }

    const { email, password } = req.body;

    const user = await User.findByEmail(email);
    if (!user) {
      sendUnauthorized(res, 'Invalid email or password');
      return;
    }

    const isPasswordValid = await user.comparePassword(password);
    if (!isPasswordValid) {
      sendUnauthorized(res, 'Invalid email or password');
      return;
    }

    const { accessToken, refreshToken } = generateTokenPair(
      user._id.toString(),
      user.email,
      user.role
    );

    // Rotate refresh tokens — keep a maximum of 5
    const updatedTokens = [...(user.refreshTokens || []), refreshToken].slice(-5);
    await User.findByIdAndUpdate(user._id, { refreshTokens: updatedTokens });

    res.cookie('refreshToken', refreshToken, getRefreshTokenCookieOptions());

    logger.info(`User logged in: ${user.email}`);

    sendSuccess(res, {
      user: { id: user._id, name: user.name, email: user.email, role: user.role },
      accessToken,
    }, 'Login successful');
  } catch (error) {
    next(error);
  }
};

export const logout = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const token = req.cookies?.refreshToken as string | undefined;

    if (token) {
      try {
        const decoded = verifyRefreshToken(token);
        await User.findByIdAndUpdate(decoded.userId, {
          $pull: { refreshTokens: token },
        });
      } catch {
        // Token may be expired — still clear cookie
      }
    }

    res.clearCookie('refreshToken', { path: '/api/auth' });
    sendSuccess(res, null, 'Logged out successfully');
  } catch (error) {
    next(error);
  }
};

export const refreshToken = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const token = req.cookies?.refreshToken as string | undefined;

    if (!token) {
      sendUnauthorized(res, 'No refresh token provided');
      return;
    }

    let decoded;
    try {
      decoded = verifyRefreshToken(token);
    } catch {
      sendUnauthorized(res, 'Invalid or expired refresh token');
      return;
    }

    const user = await User.findById(decoded.userId).select('+refreshTokens');
    if (!user || !user.refreshTokens?.includes(token)) {
      sendUnauthorized(res, 'Refresh token has been revoked');
      return;
    }

    // Rotate: remove old, add new
    const { accessToken, refreshToken: newRefreshToken } = generateTokenPair(
      user._id.toString(),
      user.email,
      user.role
    );

    const updatedTokens = user.refreshTokens
      .filter((t) => t !== token)
      .concat(newRefreshToken)
      .slice(-5);

    await User.findByIdAndUpdate(user._id, { refreshTokens: updatedTokens });

    res.cookie('refreshToken', newRefreshToken, getRefreshTokenCookieOptions());

    sendSuccess(res, { accessToken }, 'Token refreshed successfully');
  } catch (error) {
    next(error);
  }
};
