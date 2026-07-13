import { body } from 'express-validator';

export const predictionValidator = [
  body('smiles')
    .trim()
    .notEmpty()
    .withMessage('SMILES string is required')
    .isLength({ min: 1, max: 10000 })
    .withMessage('SMILES string must be between 1 and 10000 characters')
    .matches(/^[A-Za-z0-9@+\-\[\]().=#\\/%$!*:]+$/)
    .withMessage('Invalid SMILES string format'),
];
