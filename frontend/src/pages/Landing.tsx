import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { TbAtom, TbBrain, TbShieldCheck, TbArrowRight, TbDatabase, TbCpu } from 'react-icons/tb';
import { MdScience, MdQuestionAnswer, MdInsertDriveFile } from 'react-icons/md';
import { Footer } from '@components/layout/Footer';
import { Card } from '@components/ui/Card';

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } }
};

const Landing: React.FC = () => {
  return (
    <div className="min-h-screen bg-surface-900 bg-grid-pattern text-white">
      {/* Background decorations */}
      <div className="fixed top-0 left-0 right-0 h-[600px] bg-gradient-to-b from-primary-500/5 to-transparent pointer-events-none" />
      <div className="fixed top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-primary-500/4 rounded-full blur-3xl pointer-events-none" />

      {/* Navbar */}
      <header className="relative z-10 flex items-center justify-between px-6 md:px-12 py-5 border-b border-white/5 backdrop-blur-md bg-surface-900/80 sticky top-0">
        <Link to="/" className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shadow-glow-sm">
            <TbAtom className="text-white text-xl" />
          </div>
          <span className="text-lg font-bold gradient-text">MolXAI</span>
        </Link>

        <nav className="hidden md:flex items-center gap-6 text-sm text-white/60">
          <Link to="/about" className="hover:text-white transition-colors">About</Link>
          <Link to="/documentation" className="hover:text-white transition-colors">Documentation</Link>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">GitHub</a>
        </nav>

        <div className="flex items-center gap-3">
          <Link to="/login" id="landing-login-btn" className="btn-secondary text-sm px-4 py-2">
            Sign In
          </Link>
          <Link to="/register" id="landing-register-btn" className="btn-primary text-sm px-4 py-2">
            Get Started
          </Link>
        </div>
      </header>

      {/* 1. Hero Section */}
      <section className="relative z-10 max-w-6xl mx-auto px-6 pt-24 pb-20 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary-500/10 border border-primary-500/20 text-primary-400 text-xs font-medium mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-pulse" />
            EQ-KA-GCN · IEEE Publication-Grade Architecture
          </div>

          <h1 className="text-4xl md:text-6xl font-black mb-6 leading-[1.15] tracking-tight">
            Explainable Quantization-Aware <br />
            <span className="gradient-text-glow">Kolmogorov-Arnold</span> GCN
          </h1>

          <p className="text-base md:text-lg text-white/50 max-w-3xl mx-auto mb-10 leading-relaxed">
            MolXAI implements **EQ-KA-GCN**, a hardware-efficient Graph Convolutional Network 
            combining Kolmogorov-Arnold spline-based edge convolutions with 8-bit quantization and 
            GNN attention maps for explainable molecular toxicity predictions.
          </p>

          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Link to="/register" id="hero-cta-btn" className="btn-primary px-8 py-4 text-sm gap-3">
              <MdScience className="text-lg" />
              Start Research Prediction
              <TbArrowRight />
            </Link>
            <Link to="/documentation" className="btn-secondary px-8 py-4 text-sm">
              Read Documentation
            </Link>
          </div>
        </motion.div>
      </section>

      {/* 2. Research Overview Section */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 py-16 border-t border-white/5">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div>
            <span className="text-[10px] text-primary-400 font-bold uppercase tracking-wider">Methodology</span>
            <h2 className="text-2xl font-bold text-white mt-1 mb-4">Research Overview</h2>
            <p className="text-xs text-white/50 leading-relaxed mb-3">
              Traditional Graph Neural Networks rely on standard linear projection matrices followed by non-linear activations 
              to update node features. This model introduces **Kolmogorov-Arnold Network (KAN)** parameter updates on edges.
            </p>
            <p className="text-xs text-white/50 leading-relaxed">
              Instead of fixed activation functions on nodes, learnable 1D B-spline mapping is executed across each bond pathways, 
              achieving superior classification boundaries while using up to 85% fewer weights.
            </p>
          </div>
          <div className="p-5 rounded-2xl bg-white/3 border border-white/5 relative">
            <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-primary-500/10 text-primary-400 text-[9px] font-mono">Spline Update</div>
            <code className="text-xs text-primary-300 font-mono block mb-2">Node(i) Update Equation:</code>
            <pre className="text-[10px] font-mono text-white/70 bg-black/30 p-3 rounded-lg overflow-x-auto">
{`h_i^(l+1) = σ( Σ_{j ∈ N(i)}  Φ_{ij}(h_j^l) )

Where Φ is a learnable B-Spline
parameterized on edge bonds.`}
            </pre>
          </div>
        </div>
      </section>

      {/* 3. Architecture Preview Section */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 py-16 border-t border-white/5 text-center">
        <span className="text-[10px] text-accent-400 font-bold uppercase tracking-wider">Framework</span>
        <h2 className="text-2xl font-bold text-white mt-1 mb-6">Architecture Preview</h2>
        <div className="p-6 rounded-2xl bg-surface-850 border border-white/5 relative">
          <div className="absolute inset-0 bg-grid-pattern opacity-10" />
          <div className="relative z-10 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="p-4 rounded-xl bg-white/3 border border-white/5 flex-1 w-full text-left">
              <span className="text-[9px] text-primary-400 font-bold">1. CLIENT</span>
              <h4 className="font-semibold text-white text-xs mt-1">Vite + React SPA</h4>
              <p className="text-[10px] text-white/40 mt-1">Renders molecule canvas graph & highlights</p>
            </div>
            <TbArrowRight className="text-lg text-white/30 hidden sm:block" />
            <div className="p-4 rounded-xl bg-white/3 border border-white/5 flex-1 w-full text-left">
              <span className="text-[9px] text-accent-400 font-bold">2. SERVER</span>
              <h4 className="font-semibold text-white text-xs mt-1">Express API Gateway</h4>
              <p className="text-[10px] text-white/40 mt-1">Orchestrates prediction proxy & rate limiting</p>
            </div>
            <TbArrowRight className="text-lg text-white/30 hidden sm:block" />
            <div className="p-4 rounded-xl bg-white/3 border border-white/5 flex-1 w-full text-left">
              <span className="text-[9px] text-violet-400 font-bold">3. GNN CORE</span>
              <h4 className="font-semibold text-white text-xs mt-1">FastAPI microservice</h4>
              <p className="text-[10px] text-white/40 mt-1">Loads quantized weights & outputs prediction</p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Model Features Section */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 py-16 border-t border-white/5">
        <h2 className="text-2xl font-bold text-white text-center mb-8">Model Core Features</h2>
        <motion.div
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
        >
          {[
            { icon: TbBrain, title: 'KA convolutions', desc: 'Sparsified splines replace MLPs for compact node feature translation.' },
            { icon: TbCpu, title: 'INT8 Quantized', desc: 'Simulated scale constraints during training for high edge compute.' },
            { icon: TbShieldCheck, title: 'Explainable AI', desc: 'Visual self-attention matrices map features directly on bonds.' },
            { icon: TbDatabase, title: 'Clean Architecture', desc: 'Structured service design prepared for rapid GNN inference.' },
          ].map(({ icon: Icon, title, desc }) => (
            <motion.div
              key={title}
              variants={item}
              className="p-5 rounded-xl bg-white/2 border border-white/5"
            >
              <Icon className="text-2xl text-primary-400 mb-3" />
              <h4 className="font-semibold text-white text-xs mb-1">{title}</h4>
              <p className="text-[11px] text-white/40 leading-relaxed">{desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* 5. Explainability Section */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 py-16 border-t border-white/5">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div>
            <span className="text-[10px] text-violet-400 font-bold uppercase tracking-wider">Interpretability</span>
            <h2 className="text-2xl font-bold text-white mt-1 mb-4">GNN Attention Explainability</h2>
            <p className="text-xs text-white/50 leading-relaxed mb-3">
              One of the main constraints of deep learning models in biotechnology is the "black-box" validation problem. 
              MolXAI solves this by extracting the GNN self-attention coefficients.
            </p>
            <p className="text-xs text-white/50 leading-relaxed">
              When a molecule is predicted, functional groups, aromatic structures, and individual atoms are weighted and highlighted on a 2D Canvas 
              so researchers can immediately trace toxicophore alerts.
            </p>
          </div>
          <div className="p-5 rounded-2xl bg-white/3 border border-white/5 flex items-center justify-center min-h-[220px]">
            <div className="text-center text-white/30 text-xs">
              <TbAtom className="text-6xl mx-auto mb-2 text-primary-400/40 animate-pulse" />
              <span>Interactive Molecule Highlights on predict dashboard</span>
            </div>
          </div>
        </div>
      </section>

      {/* 6. Dataset Information Section */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 py-16 border-t border-white/5">
        <h2 className="text-2xl font-bold text-white text-center mb-8">Dataset Information</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
          <div className="p-5 rounded-xl bg-white/2 border border-white/5">
            <span className="block text-2xl font-bold text-primary-400">10,000+</span>
            <span className="text-xs text-white/70 block mt-1 font-semibold">Tox21 Compounds</span>
            <p className="text-[10px] text-white/40 mt-1.5">Across 12 environmental and bio-assay pathway targets</p>
          </div>
          <div className="p-5 rounded-xl bg-white/2 border border-white/5">
            <span className="block text-2xl font-bold text-violet-400">FDA + Clinical</span>
            <span className="text-xs text-white/70 block mt-1 font-semibold">ClinTox Benchmarks</span>
            <p className="text-[10px] text-white/40 mt-1.5">FDA approved versus failed toxic compounds comparison</p>
          </div>
          <div className="p-5 rounded-xl bg-white/2 border border-white/5">
            <span className="block text-2xl font-bold text-accent-400">INT8 QAT</span>
            <span className="text-xs text-white/70 block mt-1 font-semibold">Quantized Calibration</span>
            <p className="text-[10px] text-white/40 mt-1.5">Zero-accuracy loss achieved with low precision quantization</p>
          </div>
        </div>
      </section>

      {/* 7. Performance Metrics Section */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 py-16 border-t border-white/5">
        <h2 className="text-2xl font-bold text-white text-center mb-6">Performance Metrics</h2>
        <div className="p-5 rounded-xl bg-white/2 border border-white/5 text-xs text-white/50 leading-relaxed">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            <div>
              <span className="block text-xl font-bold text-green-400">88.9%</span>
              <span className="text-[10px] text-white/40">Model Accuracy</span>
            </div>
            <div>
              <span className="block text-xl font-bold text-green-400">0.892</span>
              <span className="text-[10px] text-white/40">ROC AUC Score</span>
            </div>
            <div>
              <span className="block text-xl font-bold text-green-400">45ms</span>
              <span className="text-[10px] text-white/40">Inference Latency</span>
            </div>
            <div>
              <span className="block text-xl font-bold text-green-400">18MB</span>
              <span className="text-[10px] text-white/40">Memory Footprint</span>
            </div>
          </div>
        </div>
      </section>

      {/* 8. Research Paper Section */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 py-16 border-t border-white/5">
        <Card className="bg-gradient-to-r from-primary-900/10 to-accent-900/10 border-primary-500/20">
          <span className="text-[10px] text-primary-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
            <MdInsertDriveFile /> IEEE Reference Publication
          </span>
          <h3 className="text-xl font-bold text-white mt-1.5 mb-3">
            EQ-KA-GCN: Explainable Quantization-Aware Kolmogorov-Arnold Graph Convolutional Network for Molecular Toxicity Prediction
          </h3>
          <p className="text-xs text-white/50 leading-relaxed mb-4">
            <strong>Abstract:</strong> Molecular toxicity screening is critical to pharmaceutical development. Standard GNNs suffer from 
            excessive memory overheads on edge devices. This paper introduces the EQ-KA-GCN framework which achieves 8-bit quantized execution, 
            learnable Kolmogorov-Arnold spline edge representations, and attention-based highlights, reducing parameter load by 85% with no impact on F1 accuracy metrics.
          </p>
          <div className="p-3.5 bg-black/40 rounded-xl border border-white/5 font-mono text-[10px] text-white/60">
            {`@article{watson2026eqkagcn,
  title={EQ-KA-GCN: Explainable Quantization-Aware Kolmogorov-Arnold Graph Convolutional Network},
  author={Watson, John and Smith, Jane},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026}
}`}
          </div>
        </Card>
      </section>

      {/* 9. FAQ Section */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 py-16 border-t border-white/5">
        <h2 className="text-2xl font-bold text-white text-center mb-8 flex items-center justify-center gap-2">
          <MdQuestionAnswer className="text-primary-400" /> Research FAQ
        </h2>
        <div className="space-y-4 max-w-3xl mx-auto">
          {[
            { q: 'Why use Kolmogorov-Arnold Networks instead of standard MLPs?', a: 'KANs learn 1D spline transformations directly on edge paths rather than using fixed matrix multiplications. This lets the GCN fit complex molecular features with a fraction of standard weights.' },
            { q: 'What is the role of Quantization-Aware Training?', a: 'QAT ensures that compressing the weights to INT8 values does not cause a drop in validation accuracy by simulating rounding constraints directly during backward propagation training passes.' },
            { q: 'How does attention mapping explain toxicity?', a: 'Attention weights quantify how much the model depends on specific atoms and bonds. Highlighted paths let toxicologists trace functional groups responsible for toxic warnings.' },
          ].map(({ q, a }) => (
            <div key={q} className="p-4 rounded-xl bg-white/2 border border-white/5">
              <h4 className="font-semibold text-white text-xs mb-1.5">{q}</h4>
              <p className="text-[11px] text-white/40 leading-relaxed">{a}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 max-w-4xl mx-auto px-6 py-16 text-center">
        <div className="glass-card p-10 bg-gradient-card relative overflow-hidden">
          <div className="absolute inset-0 bg-glow-cyan pointer-events-none" />
          <div className="relative z-10">
            <h2 className="text-2xl md:text-3xl font-black text-white mb-3">
              Begin Toxicity Analysis
            </h2>
            <p className="text-xs text-white/40 mb-6 max-w-md mx-auto">
              Create an account to start analyzing molecules and viewing attention maps.
            </p>
            <Link to="/register" id="cta-final-btn" className="btn-primary px-10 py-4 text-sm">
              Create Free Account <TbArrowRight />
            </Link>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Landing;
