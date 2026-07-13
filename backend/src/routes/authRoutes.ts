import { Router } from 'express';
import cookieParser from 'cookie-parser';
import * as authController from '../controllers/authController';
import { registerValidator, loginValidator } from '../validators/authValidator';
import { authRateLimiter } from '../middleware/rateLimiter';

const router = Router();

router.use(cookieParser());

router.post('/register', authRateLimiter, registerValidator, authController.register);
router.post('/login', authRateLimiter, loginValidator, authController.login);
router.post('/logout', authController.logout);
router.post('/refresh', authController.refreshToken);

export default router;
