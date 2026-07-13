import React from 'react';
import { ImSpinner8 } from 'react-icons/im';

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  text?: string;
  fullPage?: boolean;
}

const sizeClasses = {
  sm: 'text-base',
  md: 'text-2xl',
  lg: 'text-4xl',
  xl: 'text-6xl',
};

export const Spinner: React.FC<SpinnerProps> = ({
  size = 'md',
  className = '',
  text,
  fullPage = false,
}) => {
  const spinner = (
    <div className={`flex flex-col items-center justify-center gap-3 ${className}`}>
      <ImSpinner8 className={`animate-spin text-primary-400 ${sizeClasses[size]}`} />
      {text && <p className="text-sm text-white/50 animate-pulse">{text}</p>}
    </div>
  );

  if (fullPage) {
    return (
      <div className="fixed inset-0 bg-surface-900/80 backdrop-blur-sm flex items-center justify-center z-50">
        {spinner}
      </div>
    );
  }

  return spinner;
};
