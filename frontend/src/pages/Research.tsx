import React from 'react';
import { Card } from '@components/ui/Card';
import { TbBrain, TbDatabase, TbGitBranch, TbTarget, TbHourglass } from 'react-icons/tb';

export const Research: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-white">Research Overview</h1>
        <p className="text-white/40 text-sm mt-1">
          Detailed overview of the EQ-KA-GCN research paper and GNN model formulation.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Objective */}
        <Card>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shrink-0">
              <TbTarget className="text-white text-xl" />
            </div>
            <h2 className="text-base font-semibold text-white">Project Objective</h2>
          </div>
          <p className="text-xs text-white/50 leading-relaxed">
            The primary goal of EQ-KA-GCN is to develop an explainable, highly accurate, and hardware-efficient 
            Graph Convolutional Network utilizing Kolmogorov-Arnold Networks (KAN) instead of traditional Multi-Layer Perceptrons (MLPs). 
            By implementing Quantization-Aware Training (QAT), the model achieves lightweight edge-device compatibility 
            for real-time molecular toxicity screening during drug discovery.
          </p>
        </Card>

        {/* Dataset */}
        <Card>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shrink-0">
              <TbDatabase className="text-white text-xl" />
            </div>
            <h2 className="text-base font-semibold text-white">Datasets Utilized</h2>
          </div>
          <div className="space-y-3 text-xs text-white/50">
            <p>
              The GNN model is trained and evaluated using well-established public benchmarking datasets:
            </p>
            <ul className="space-y-2 pl-4 list-disc">
              <li><strong>Tox21:</strong> Toxicity assessment of 10,000+ environmental chemicals and drugs across 12 nuclear receptor pathways.</li>
              <li><strong>ClinTox:</strong> Comparison of FDA-approved drugs versus clinical trials failures due to toxic side-effects.</li>
              <li><strong>SIDER:</strong> Side Effect Resource documenting drug adverse reactions mapped to therapeutic indications.</li>
            </ul>
          </div>
        </Card>
      </div>

      {/* Methodology */}
      <Card>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shrink-0">
            <TbBrain className="text-white text-xl" />
          </div>
          <h2 className="text-base font-semibold text-white">Methodology & EQ-KA-GCN Core Logic</h2>
        </div>
        <div className="space-y-4 text-xs text-white/50 leading-relaxed">
          <p>
            In traditional GCNs, node representation updates are mapped using standard linear projections followed by 
            fixed activation functions. In contrast, <strong>EQ-KA-GCN</strong> replaces traditional weight matrices with 
            learnable 1D B-spline activation functions on the graph edges, based on the Kolmogorov-Arnold representation theorem.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            <div className="p-3.5 rounded-xl bg-white/3 border border-white/5">
              <h4 className="font-semibold text-white mb-1.5">Kolmogorov-Arnold Update</h4>
              <p className="text-[11px] opacity-80">
                Replaces standard matrix multiplication $W \cdot x$ with a sum of univariate activation functions $\sum \Phi(x)$, capturing complex molecular topologies with far fewer parameters.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-white/3 border border-white/5">
              <h4 className="font-semibold text-white mb-1.5">Quantization-Aware Training (QAT)</h4>
              <p className="text-[11px] opacity-80">
                Simulates low-precision 8-bit integer quantization during model training, ensuring zero degradation in classification accuracy when deployed to resource-constrained systems.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-white/3 border border-white/5">
              <h4 className="font-semibold text-white mb-1.5">GNN Attention Maps</h4>
              <p className="text-[11px] opacity-80">
                Self-attention mechanisms weight the relevance of chemical bonds and functional groups (e.g. aromatic rings, halogens) to explain exactly why a molecule is toxic.
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Pipeline & Future Work */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <Card>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center shrink-0">
              <TbGitBranch className="text-white text-xl" />
            </div>
            <h2 className="text-base font-semibold text-white">Pipeline Workflow</h2>
          </div>
          <div className="space-y-2 text-xs text-white/50">
            <p>1. <strong>Input:</strong> Raw SMILES string is parsed and validated.</p>
            <p>2. <strong>Graph construction:</strong> Atoms mapped to nodes; chemical bonds mapped to edges.</p>
            <p>3. <strong>EQ-KA-GCN Layers:</strong> Kolmogorov-Arnold convolutions propagate spatial and geometric features.</p>
            <p>4. <strong>Attention Extractor:</strong> Evaluates saliency scores for explainable visualization.</p>
            <p>5. <strong>Quantization Output:</strong> 8-bit prediction mapping toxicity classification output.</p>
          </div>
        </Card>

        <Card>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-pink-500 to-rose-500 flex items-center justify-center shrink-0">
              <TbHourglass className="text-white text-xl" />
            </div>
            <h2 className="text-base font-semibold text-white">Expected Results & Future Work</h2>
          </div>
          <p className="text-xs text-white/50 leading-relaxed mb-3">
            The platform is expected to achieve comparable state-of-the-art results on Tox21 dataset classifications 
            with up to a 60% reduction in model size and inference memory overhead. 
          </p>
          <p className="text-xs text-white/50 leading-relaxed">
            Future work includes extending the equivariant convolutions to 3D conformation graphs and integrating 
            generative molecule creation pipelines using variational autoencoders.
          </p>
        </Card>
      </div>
    </div>
  );
};

export default Research;
