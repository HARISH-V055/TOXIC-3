import React from 'react';
import { motion } from 'framer-motion';
import { BsCheckCircleFill, BsExclamationTriangleFill, BsInfoCircleFill, BsXCircleFill } from 'react-icons/bs';

type AlertType = 'success' | 'error' | 'warning' | 'info';

interface AlertProps {
  type: AlertType;
  message: string;
  title?: string;
  onClose?: () => void;
  className?: string;
}

const alertConfig = {
  success: {
    icon: BsCheckCircleFill,
    classes: 'bg-green-500/10 border-green-500/20 text-green-400',
    iconClass: 'text-green-400',
  },
  error: {
    icon: BsXCircleFill,
    classes: 'bg-red-500/10 border-red-500/20 text-red-400',
    iconClass: 'text-red-400',
  },
  warning: {
    icon: BsExclamationTriangleFill,
    classes: 'bg-yellow-500/10 border-yellow-500/20 text-yellow-400',
    iconClass: 'text-yellow-400',
  },
  info: {
    icon: BsInfoCircleFill,
    classes: 'bg-primary-500/10 border-primary-500/20 text-primary-300',
    iconClass: 'text-primary-400',
  },
};

export const Alert: React.FC<AlertProps> = ({ type, message, title, onClose, className = '' }) => {
  const config = alertConfig[type];
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className={`flex items-start gap-3 px-4 py-3 rounded-xl border ${config.classes} ${className}`}
    >
      <Icon className={`shrink-0 mt-0.5 text-base ${config.iconClass}`} />
      <div className="flex-1 min-w-0">
        {title && <p className="font-semibold text-sm mb-0.5">{title}</p>}
        <p className="text-sm opacity-90">{message}</p>
      </div>
      {onClose && (
        <button
          onClick={onClose}
          className="shrink-0 text-current opacity-60 hover:opacity-100 transition-opacity"
          aria-label="Close"
        >
          ×
        </button>
      )}
    </motion.div>
  );
};
