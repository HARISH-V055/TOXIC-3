import React from 'react';
import { motion } from 'framer-motion';
import { Card } from '@components/ui/Card';

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  trend?: { value: number; label: string };
  iconBg?: string;
  delay?: number;
}

export const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  trend,
  iconBg = 'from-primary-500 to-accent-500',
  delay = 0,
}) => {
  return (
    <Card className="stats-card" delay={delay}>
      <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${iconBg} flex items-center justify-center shrink-0 shadow-glow-sm`}>
        <span className="text-white text-xl">{icon}</span>
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-white/50 mb-1">{title}</p>
        <motion.p
          className="text-2xl font-bold text-white"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4, delay: delay + 0.2 }}
        >
          {value}
        </motion.p>
        {subtitle && <p className="text-xs text-white/35 mt-0.5">{subtitle}</p>}
        {trend && (
          <div className={`flex items-center gap-1 mt-1 text-xs font-medium ${trend.value >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            <span>{trend.value >= 0 ? '↑' : '↓'}</span>
            <span>{Math.abs(trend.value)}% {trend.label}</span>
          </div>
        )}
      </div>
    </Card>
  );
};
