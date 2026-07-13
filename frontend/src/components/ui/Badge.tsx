import React from 'react';
import { PredictionResult } from '@/types';

interface BadgeProps {
  children?: React.ReactNode;
  variant?: 'toxic' | 'safe' | 'pending' | 'info' | 'warning';
  prediction?: PredictionResult;
  className?: string;
}

const variantClasses = {
  toxic: 'badge-toxic',
  safe: 'badge-safe',
  pending: 'badge-pending',
  info: 'badge-info',
  warning: 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-yellow-500/15 text-yellow-400 border border-yellow-500/20',
};

const predictionToVariant = (prediction: PredictionResult): BadgeProps['variant'] => {
  switch (prediction) {
    case 'toxic': return 'toxic';
    case 'non-toxic': return 'safe';
    case 'pending': return 'pending';
    case 'error': return 'warning';
    default: return 'info';
  }
};

const predictionDots = {
  toxic: 'bg-red-400',
  safe: 'bg-green-400',
  pending: 'bg-yellow-400',
  warning: 'bg-orange-400',
  info: 'bg-primary-400',
};

export const Badge: React.FC<BadgeProps> = ({ children, variant, prediction, className = '' }) => {
  const resolvedVariant: NonNullable<BadgeProps['variant']> = prediction ? predictionToVariant(prediction) ?? 'info' : variant ?? 'info';
  const label = prediction === 'non-toxic' ? 'Non-Toxic' : prediction ? prediction.charAt(0).toUpperCase() + prediction.slice(1) : children;

  return (
    <span className={`${variantClasses[resolvedVariant]} ${className}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${predictionDots[resolvedVariant] ?? 'bg-primary-400'}`} />
      {label}
    </span>
  );
};
