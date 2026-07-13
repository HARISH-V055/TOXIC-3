import React from 'react';
import { Card } from '@components/ui/Card';
import { TbDeviceAnalytics, TbTimeline } from 'react-icons/tb';

export const Experiments: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-white">Experimental Results</h1>
        <p className="text-white/40 text-sm mt-1">
          Visual validation curves, confusion matrices, and metrics from the EQ-KA-GCN training logs.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Model Accuracy & Loss Curves */}
        <Card>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shrink-0">
              <TbTimeline className="text-white text-xl" />
            </div>
            <h2 className="text-base font-semibold text-white">Training Loss & Validation Accuracy</h2>
          </div>
          <div className="relative h-60 bg-surface-850 rounded-xl border border-white/5 p-4 flex flex-col justify-between">
            <div className="absolute inset-0 bg-grid-pattern opacity-10" />
            
            {/* SVG Graph */}
            <svg viewBox="0 0 400 200" className="w-full h-full relative z-10 overflow-visible">
              {/* Grid Lines */}
              <line x1="0" y1="180" x2="400" y2="180" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              <line x1="0" y1="135" x2="400" y2="135" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              <line x1="0" y1="90" x2="400" y2="90" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              <line x1="0" y1="45" x2="400" y2="45" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              <line x1="0" y1="10" x2="400" y2="10" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />

              {/* Loss Curve (Downwards) */}
              <path
                d="M 10,160 Q 100,100 200,60 T 400,28"
                fill="none"
                stroke="#06b6d4"
                strokeWidth="2.5"
              />
              {/* Accuracy Curve (Upwards) */}
              <path
                d="M 10,170 Q 100,90 200,45 T 400,18"
                fill="none"
                stroke="#a855f7"
                strokeWidth="2"
                strokeDasharray="4 3"
              />
            </svg>
            
            {/* Labels */}
            <div className="flex justify-between items-center text-[10px] text-white/30 px-2">
              <span>Epoch 0</span>
              <span>Epoch 50</span>
              <span>Epoch 100</span>
              <span>Epoch 150</span>
              <span>Epoch 200</span>
            </div>
            
            {/* Legend */}
            <div className="flex gap-4 text-[10px] justify-center mt-2 border-t border-white/5 pt-2">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 bg-cyan-400 inline-block" /> Training Loss</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 border-t-2 border-dashed border-purple-500 inline-block" /> Validation Accuracy</span>
            </div>
          </div>
        </Card>

        {/* ROC AUC & PR Curves */}
        <Card>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shrink-0">
              <TbTimeline className="text-white text-xl" />
            </div>
            <h2 className="text-base font-semibold text-white">ROC & Precision-Recall Curve</h2>
          </div>
          <div className="relative h-60 bg-surface-850 rounded-xl border border-white/5 p-4 flex flex-col justify-between">
            <div className="absolute inset-0 bg-grid-pattern opacity-10" />

            {/* SVG Graph */}
            <svg viewBox="0 0 400 200" className="w-full h-full relative z-10 overflow-visible">
              {/* Diagonal base line */}
              <line x1="10" y1="190" x2="390" y2="10" stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />

              {/* ROC Curve */}
              <path
                d="M 10,190 C 10,40 120,10 390,10"
                fill="none"
                stroke="#10b981"
                strokeWidth="2.5"
              />
            </svg>

            <div className="flex justify-between items-center text-[10px] text-white/30 px-2">
              <span>0.0 False Positive Rate</span>
              <span>1.0</span>
            </div>

            {/* Legend */}
            <div className="flex gap-4 text-[10px] justify-center mt-2 border-t border-white/5 pt-2">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 bg-green-400 inline-block" /> ROC Curve (AUC = 0.892)</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Confusion Matrix */}
      <Card className="max-w-md mx-auto">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shrink-0">
            <TbDeviceAnalytics className="text-white text-xl" />
          </div>
          <h2 className="text-base font-semibold text-white">Confusion Matrix (Quantized INT8)</h2>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center text-xs">
          {/* Top labels */}
          <div />
          <div className="font-semibold text-white/50">Predicted Safe</div>
          <div className="font-semibold text-white/50">Predicted Toxic</div>

          {/* Row 1 */}
          <div className="font-semibold text-white/50 flex items-center justify-center text-[11px]">Actual Safe</div>
          <div className="p-4 rounded-xl bg-green-500/10 border border-green-500/20 font-bold text-green-400">
            842 <span className="block text-[9px] font-normal text-white/35">True Negative (92.5%)</span>
          </div>
          <div className="p-4 rounded-xl bg-red-500/5 border border-red-500/10 font-bold text-red-400/70">
            68 <span className="block text-[9px] font-normal text-white/35">False Positive (7.5%)</span>
          </div>

          {/* Row 2 */}
          <div className="font-semibold text-white/50 flex items-center justify-center text-[11px]">Actual Toxic</div>
          <div className="p-4 rounded-xl bg-red-500/5 border border-red-500/10 font-bold text-red-400/70">
            42 <span className="block text-[9px] font-normal text-white/35">False Negative (5.8%)</span>
          </div>
          <div className="p-4 rounded-xl bg-green-500/10 border border-green-500/20 font-bold text-green-400">
            678 <span className="block text-[9px] font-normal text-white/35">True Positive (94.2%)</span>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Experiments;
