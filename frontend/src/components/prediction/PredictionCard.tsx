import React from 'react';
import { motion } from 'framer-motion';
import { Prediction } from '@/types';
import { Badge } from '@components/ui/Badge';
import { Card } from '@components/ui/Card';

interface ConfidenceBarProps {
  value: number;
  label?: string;
  color?: 'primary' | 'green' | 'red';
  className?: string;
}

export const ConfidenceBar: React.FC<ConfidenceBarProps> = ({
  value,
  label,
  color = 'primary',
  className = '',
}) => {
  const percentageVal = value * 100;
  const displayPercentage = percentageVal < 0.01 && percentageVal > 0 ? '<0.01' : percentageVal.toFixed(2);

  const gradients = {
    primary: 'linear-gradient(90deg, #06b6d4, #3b82f6)',
    green: 'linear-gradient(90deg, #22c55e, #16a34a)',
    red: 'linear-gradient(90deg, #ef4444, #dc2626)',
    blue: 'linear-gradient(90deg, #3b82f6, #6366f1)',
  };

  return (
    <div className={`w-full ${className}`}>
      <div className="flex items-center justify-between mb-1.5">
        {label && <span className="text-xs text-white/50">{label}</span>}
        <span className="text-xs font-semibold text-white ml-auto">{displayPercentage}%</span>
      </div>
      <div className="confidence-bar-track">
        <motion.div
          className="confidence-bar-fill"
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(Math.max(percentageVal, 0.5), 100)}%` }}
          transition={{ duration: 1, ease: 'easeOut', delay: 0.2 }}
          style={{ background: gradients[color] }}
        />
      </div>
    </div>
  );
};

interface PredictionCardProps {
  prediction: Prediction;
  compact?: boolean;
  className?: string;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  prediction,
  compact = false,
  className = '',
}) => {
  const isToxic = prediction.prediction === 'toxic';
  const isPending = prediction.prediction === 'pending';

  return (
    <Card className={className} hover>
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex-1 min-w-0">
          <p className="text-xs text-white/40 mb-1">SMILES</p>
          <code className="text-sm font-mono text-primary-300 break-all line-clamp-2">
            {prediction.smiles}
          </code>
        </div>
        <Badge prediction={prediction.prediction} />
      </div>

      {!isPending && prediction.probability !== null && (
        <div className="space-y-3">
          <ConfidenceBar
            value={prediction.probability}
            label="Toxicity Probability"
            color={isToxic ? 'red' : 'green'}
          />
          {!compact && prediction.confidence !== null && (
            <ConfidenceBar
              value={prediction.confidence}
              label="Model Confidence"
              color="primary"
            />
          )}
        </div>
      )}

      {!compact && (
        <div className="flex items-center justify-between mt-4 pt-3 border-t border-white/5 text-xs text-white/35">
          <span>
            {prediction.importantAtoms.length > 0
              ? `${prediction.importantAtoms.length} atoms highlighted`
              : 'No highlights'}
          </span>
          {(prediction.inferenceTimeMs !== null && prediction.inferenceTimeMs !== undefined) ? (
            <span>{prediction.inferenceTimeMs.toFixed(0)}ms</span>
          ) : (prediction.totalResponseTimeMs ? (
            <span>{prediction.totalResponseTimeMs.toFixed(0)}ms</span>
          ) : null)}
        </div>
      )}
    </Card>
  );
};
