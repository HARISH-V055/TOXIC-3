import { Router } from 'express';
import * as userController from '../controllers/userController';
import { authenticate } from '../middleware/auth';
import { updateProfileValidator, changePasswordValidator } from '../validators/userValidator';

const router = Router();

router.use(authenticate);

router.get('/profile', userController.getProfile);
router.put('/profile', updateProfileValidator, userController.updateProfile);
router.put('/password', changePasswordValidator, userController.changePassword);
router.delete('/', userController.deleteAccount);

export default router;
