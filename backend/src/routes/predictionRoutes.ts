import { Router } from 'express';
import * as predictionController from '../controllers/predictionController';
import { authenticate } from '../middleware/auth';
import { predictionValidator } from '../validators/predictionValidator';
import { predictionRateLimiter } from '../middleware/rateLimiter';

const router = Router();

router.use(authenticate);

router.post('/', predictionRateLimiter, predictionValidator, predictionController.createPrediction);
router.get('/', predictionController.getPredictions);
router.get('/stats', predictionController.getPredictionStats);
router.get('/:id', predictionController.getPredictionById);
router.delete('/:id', predictionController.deletePrediction);

export default router;
