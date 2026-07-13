import React from 'react';
import { Card } from '@components/ui/Card';
import { TbLayout, TbNetwork, TbArrowRight } from 'react-icons/tb';

export const Architecture: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-white">System Architecture</h1>
        <p className="text-white/40 text-sm mt-1">
          Detailed technical blueprints mapping the EQ-KA-GCN hardware-efficient GNN service and full-stack API.
        </p>
      </div>

      {/* Main diagram card */}
      <Card>
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shrink-0">
            <TbLayout className="text-white text-xl" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">Overall System Architecture</h2>
            <p className="text-xs text-white/30">Three-tier architecture with stateless API and GNN Microservice</p>
          </div>
        </div>

        {/* Visual Map */}
        <div className="p-6 rounded-2xl bg-surface-850 border border-white/5 relative overflow-hidden">
          <div className="absolute inset-0 bg-grid-pattern opacity-10" />
          <div className="relative z-10 flex flex-col md:flex-row items-center justify-center gap-6 text-center">
            
            {/* Frontend */}
            <div className="w-full md:w-1/4 p-4 rounded-xl bg-primary-500/10 border border-primary-500/20">
              <span className="text-[10px] text-primary-400 font-bold uppercase tracking-wider">Client Layer</span>
              <h4 className="font-semibold text-white mt-1 text-sm">Vite + React 19 SPA</h4>
              <p className="text-[10px] text-white/45 mt-1.5">Tailwind CSS UI, Zustand state, Canvas Molecular viewer</p>
            </div>

            <TbArrowRight className="text-2xl text-white/20 hidden md:block" />

            {/* Backend Proxy */}
            <div className="w-full md:w-1/3 p-4 rounded-xl bg-accent-500/10 border border-accent-500/20">
              <span className="text-[10px] text-accent-400 font-bold uppercase tracking-wider">Controller Layer</span>
              <h4 className="font-semibold text-white mt-1 text-sm">Node.js Express API</h4>
              <p className="text-[10px] text-white/45 mt-1.5">JWT Token Rotation, Rate Limiter, Mongoose MongoDB history database</p>
            </div>

            <TbArrowRight className="text-2xl text-white/20 hidden md:block" />

            {/* AI Service */}
            <div className="w-full md:w-1/4 p-4 rounded-xl bg-violet-500/10 border border-violet-500/20">
              <span className="text-[10px] text-violet-400 font-bold uppercase tracking-wider">Inference Layer</span>
              <h4 className="font-semibold text-white mt-1 text-sm">Python FastAPI Service</h4>
              <p className="text-[10px] text-white/45 mt-1.5">EQ-KA-GCN PyTorch Geometric model, QAT quantized inference</p>
            </div>

          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Model Pipeline */}
        <Card>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shrink-0">
              <TbNetwork className="text-white text-xl" />
            </div>
            <h2 className="text-base font-semibold text-white">EQ-KA-GCN AI Pipeline</h2>
          </div>
          <div className="space-y-4">
            <p className="text-xs text-white/50 leading-relaxed">
              Inside the python microservice, molecular graphs are compiled dynamically using the RDKit library and 
              processed through the Equivariant Convolution layers.
            </p>
            <div className="space-y-2 text-xs text-white/40">
              <div className="flex items-center gap-2.5 p-2 rounded-lg bg-white/3 border border-white/5">
                <span className="w-5 h-5 rounded bg-white/5 flex items-center justify-center text-[10px] text-primary-300 font-bold">1</span>
                <span>SMILES input string normalization & atom parsing</span>
              </div>
              <div className="flex items-center gap-2.5 p-2 rounded-lg bg-white/3 border border-white/5">
                <span className="w-5 h-5 rounded bg-white/5 flex items-center justify-center text-[10px] text-primary-300 font-bold">2</span>
                <span>Construct node matrices (atomic numbers, formal charge, chirality)</span>
              </div>
              <div className="flex items-center gap-2.5 p-2 rounded-lg bg-white/3 border border-white/5">
                <span className="w-5 h-5 rounded bg-white/5 flex items-center justify-center text-[10px] text-primary-300 font-bold">3</span>
                <span>Kolmogorov-Arnold convolution parameter update & Spline interpolation</span>
              </div>
              <div className="flex items-center gap-2.5 p-2 rounded-lg bg-white/3 border border-white/5">
                <span className="w-5 h-5 rounded bg-white/5 flex items-center justify-center text-[10px] text-primary-300 font-bold">4</span>
                <span>Attention-weighted graph aggregation mapping</span>
              </div>
            </div>
          </div>
        </Card>

        {/* Quantization Engine */}
        <Card>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shrink-0">
              <TbNetwork className="text-white text-xl" />
            </div>
            <h2 className="text-base font-semibold text-white">Quantization-Aware Framework</h2>
          </div>
          <div className="space-y-3 text-xs text-white/50 leading-relaxed">
            <p>
              Quantization-Aware Training (QAT) models the effects of low-precision integer calculations during 
              the forward and backward passes using straight-through estimators (STE).
            </p>
            <div className="p-3.5 rounded-xl bg-white/3 border border-white/5">
              <h4 className="font-semibold text-white mb-1.5 text-xs">8-bit Quantized Mapping</h4>
              <p className="text-[11px] text-white/40">
                Weights and activations are mapped to dynamic INT8 scaling factors:
                <br />
                <code className="text-primary-300 font-mono block mt-1.5 text-[10px]">
                  X_quant = clamp(round(X / Scale) + ZeroPoint, -128, 127)
                </code>
              </p>
            </div>
            <p className="text-[11px] text-white/40">
              This layout guarantees high edge-computation performance without sacrificing toxicity categorization accuracy.
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Architecture;
