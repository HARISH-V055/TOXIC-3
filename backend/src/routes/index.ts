import { Router } from 'express';
import authRoutes from './authRoutes';
import userRoutes from './userRoutes';
import predictionRoutes from './predictionRoutes';
import aiRoutes from './aiRoutes';

const router = Router();

router.use('/auth', authRoutes);
router.use('/user', userRoutes);
router.use('/predictions', predictionRoutes);
router.use('/ai', aiRoutes);

export default router;
