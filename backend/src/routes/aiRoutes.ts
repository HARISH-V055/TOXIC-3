import { Router } from 'express';
import * as aiController from '../controllers/aiController';
import { authenticate } from '../middleware/auth';
import { predictionValidator } from '../validators/predictionValidator';
import { predictionRateLimiter } from '../middleware/rateLimiter';

const router = Router();

router.get('/status', aiController.getModelStatus);

router.use(authenticate);

router.post('/predict', predictionRateLimiter, predictionValidator, aiController.predict);
router.post('/explain', predictionRateLimiter, predictionValidator, aiController.explain);

export default router;
