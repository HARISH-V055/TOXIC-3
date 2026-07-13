import React from 'react';
import { TbBrain } from 'react-icons/tb';
import { Card } from '@components/ui/Card';
import { ModelStatus } from '@/types';

interface ModelStatusProps {
  status: ModelStatus | null;
}

export const ModelStatusCard: React.FC<ModelStatusProps> = ({ status }) => {
  const isOnline = status?.service?.status === 'online';
  const modelStatus = status?.model?.status ?? 'offline';

  return (
    <Card delay={0.3}>
      <div className="flex items-start gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shrink-0">
          <TbBrain className="text-white text-xl" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-white">Model Status</h3>
          <p className="text-xs text-white/40">{status?.model?.name ?? 'EQ-KA-GCN'}</p>
        </div>
      </div>

      <div className="space-y-2.5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-white/50">AI Service</span>
          <div className="flex items-center gap-1.5">
            <span
              className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}
            />
            <span className={isOnline ? 'text-green-400' : 'text-red-400'}>
              {isOnline ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between text-xs">
          <span className="text-white/50">Model</span>
          <span className={
            modelStatus === 'active' ? 'text-green-400' :
            modelStatus === 'training' ? 'text-yellow-400' : 'text-red-400/70'
          }>
            {modelStatus.charAt(0).toUpperCase() + modelStatus.slice(1)}
          </span>
        </div>

        <div className="flex items-center justify-between text-xs">
          <span className="text-white/50">Version</span>
          <span className="text-white/70 font-mono">{status?.model?.version ?? 'N/A'}</span>
        </div>

        {!isOnline && (
          <div className="mt-3 px-3 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
            <p className="text-xs text-yellow-400/80">
              AI service is not connected. Predictions will return placeholder results.
            </p>
          </div>
        )}
      </div>
    </Card>
  );
};
