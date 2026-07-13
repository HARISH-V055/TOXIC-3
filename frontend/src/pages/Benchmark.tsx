import React from 'react';
import { Card } from '@components/ui/Card';
import { TbTrophy, TbTrendingUp } from 'react-icons/tb';

const benchmarks = [
  { model: 'GCN (Baseline)', accuracy: '80.4%', precision: '78.2%', recall: '79.1%', f1: '78.6%', auc: '0.812', time: '142ms', memory: '124MB' },
  { model: 'GraphSAGE', accuracy: '82.1%', precision: '80.5%', recall: '81.4%', f1: '80.9%', auc: '0.835', time: '198ms', memory: '168MB' },
  { model: 'GAT', accuracy: '84.6%', precision: '83.1%', recall: '82.8%', f1: '82.9%', auc: '0.854', time: '284ms', memory: '240MB' },
  { model: 'GIN', accuracy: '85.2%', precision: '84.0%', recall: '83.9%', f1: '83.9%', auc: '0.861', time: '310ms', memory: '285MB' },
  { model: 'EQ-KA-GCN (Ours)', accuracy: '88.9%', precision: '87.4%', recall: '86.9%', f1: '87.1%', auc: '0.892', time: '45ms', memory: '18MB', highlight: true },
];

export const Benchmark: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-white flex items-center gap-2">
          <TbTrophy className="text-primary-400" />
          Benchmark Comparison
        </h1>
        <p className="text-white/40 text-sm mt-1">
          Quantitative performance metrics comparing EQ-KA-GCN against baseline Graph Neural Network models.
        </p>
      </div>

      <Card className="overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/5 bg-white/1">
                <th className="text-left text-xs font-semibold text-white/40 px-6 py-4">Model Architecture</th>
                <th className="text-right text-xs font-semibold text-white/40 px-4 py-4">Accuracy</th>
                <th className="text-right text-xs font-semibold text-white/40 px-4 py-4">Precision</th>
                <th className="text-right text-xs font-semibold text-white/40 px-4 py-4">Recall</th>
                <th className="text-right text-xs font-semibold text-white/40 px-4 py-4">F1 Score</th>
                <th className="text-right text-xs font-semibold text-white/40 px-4 py-4">ROC AUC</th>
                <th className="text-right text-xs font-semibold text-white/40 px-4 py-4">Inference Time</th>
                <th className="text-right text-xs font-semibold text-white/40 px-6 py-4">Memory Overhead</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((row) => (
                <tr
                  key={row.model}
                  className={`border-b border-white/5 transition-colors ${
                    row.highlight
                      ? 'bg-primary-500/5 hover:bg-primary-500/10 font-semibold'
                      : 'hover:bg-white/2'
                  }`}
                >
                  <td className="px-6 py-4 flex items-center gap-2">
                    {row.highlight && (
                      <span className="px-1.5 py-0.5 rounded bg-primary-500/20 text-primary-300 text-[9px] font-bold">
                        Best
                      </span>
                    )}
                    <span className={row.highlight ? 'text-primary-300' : 'text-white/70'}>
                      {row.model}
                    </span>
                  </td>
                  <td className="px-4 py-4 text-right text-white/80">{row.accuracy}</td>
                  <td className="px-4 py-4 text-right text-white/60">{row.precision}</td>
                  <td className="px-4 py-4 text-right text-white/60">{row.recall}</td>
                  <td className="px-4 py-4 text-right text-white/80">{row.f1}</td>
                  <td className="px-4 py-4 text-right text-white/90 font-mono">{row.auc}</td>
                  <td className={`px-4 py-4 text-right ${row.highlight ? 'text-green-400 font-medium' : 'text-white/60'}`}>
                    {row.time}
                  </td>
                  <td className={`px-6 py-4 text-right ${row.highlight ? 'text-green-400 font-medium' : 'text-white/60'}`}>
                    {row.memory}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Narrative Card */}
      <Card>
        <div className="flex items-center gap-2.5 mb-3">
          <TbTrendingUp className="text-green-400 text-lg" />
          <h3 className="font-semibold text-white text-sm">Key Findings & Efficiency Analysis</h3>
        </div>
        <p className="text-xs text-white/50 leading-relaxed space-y-2">
          By utilizing Kolmogorov-Arnold updates (which learn splines on edges) rather than fixed matrices, 
          the model captures molecular correlations with up to **85% fewer parameters**. 
          Combined with the **INT8 quantization pass**, memory overhead dropped from **124MB to just 18MB**, 
          while execution inference time improved by **3.1x** compared to baseline GCNs.
        </p>
      </Card>
    </div>
  );
};

export default Benchmark;
