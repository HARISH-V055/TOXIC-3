import React from 'react';
import { motion } from 'framer-motion';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  onClick?: () => void;
  animate?: boolean;
  delay?: number;
}

const paddingClasses = {
  none: '',
  sm: 'p-4',
  md: 'p-6',
  lg: 'p-8',
};

export const Card: React.FC<CardProps> = ({
  children,
  className = '',
  hover = false,
  padding = 'md',
  onClick,
  animate = true,
  delay = 0,
}) => {
  return (
    <motion.div
      initial={animate ? { opacity: 0, y: 16 } : false}
      animate={animate ? { opacity: 1, y: 0 } : false}
      transition={{ duration: 0.4, delay, ease: 'easeOut' }}
      onClick={onClick}
      className={`
        ${hover ? 'glass-card-hover' : 'glass-card'}
        ${paddingClasses[padding]}
        ${onClick ? 'cursor-pointer' : ''}
        ${className}
      `}
    >
      {children}
    </motion.div>
  );
};
