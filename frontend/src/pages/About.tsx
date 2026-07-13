import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { TbAtom, TbBrain, TbCode, TbArrowRight } from 'react-icons/tb';
import { Footer } from '@components/layout/Footer';

const About: React.FC = () => {
  return (
    <div className="min-h-screen bg-surface-900 bg-grid-pattern text-white">
      <header className="border-b border-white/5 px-6 py-4 flex items-center gap-3 backdrop-blur-md bg-surface-900/80 sticky top-0 z-10">
        <Link to="/" className="flex items-center gap-2">
          <TbAtom className="text-primary-400 text-xl" />
          <span className="font-bold gradient-text">MolXAI</span>
        </Link>
        <span className="text-white/20">/</span>
        <span className="text-sm text-white/50">About</span>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-16 space-y-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl md:text-5xl font-black text-white mb-4">
            About <span className="gradient-text">EQ-KA-GCN</span>
          </h1>
          <p className="text-lg text-white/50 leading-relaxed max-w-3xl">
            MolXAI implements the **EQ-KA-GCN** (Explainable Quantization-Aware Kolmogorov-Arnold Graph Convolutional Network) 
            model to accelerate drug discovery pipelines and chemical safety profiling.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {[
            { icon: TbBrain, title: 'Kolmogorov-Arnold Convolutions', desc: 'Replaces conventional MLPs with B-splines parameterized directly on graph edges, learning precise non-linear updates with 85% fewer weights.' },
            { icon: TbAtom, title: 'Quantization-Aware Training', desc: 'Simulates 8-bit precision limits during training, enabling high-speed localized execution on edge devices with zero loss in classification F1.' },
            { icon: TbCode, title: 'Explainable Self-Attention', desc: 'Computes and returns atom-wise and bond-wise attention coefficients, highlighting potential toxicophore alerts on a 2D interactive canvas.' },
          ].map(({ icon: Icon, title, desc }) => (
            <div key={title} className="glass-card p-6">
              <Icon className="text-3xl text-primary-400 mb-4" />
              <h3 className="text-base font-semibold text-white mb-2">{title}</h3>
              <p className="text-sm text-white/40 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>

        <div className="glass-card p-8">
          <h2 className="text-2xl font-bold text-white mb-4">Technology Stack</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {['React 19', 'TypeScript', 'Tailwind CSS', 'Framer Motion', 'Node.js', 'Express.js', 'MongoDB', 'FastAPI', 'PyTorch', 'PyTorch Geometric', 'JWT Auth', 'Docker'].map((tech) => (
              <div key={tech} className="px-3 py-2 rounded-lg bg-white/5 border border-white/5 text-center text-xs text-white/60 hover:text-white hover:border-white/10 transition-colors">
                {tech}
              </div>
            ))}
          </div>
        </div>

        <div className="text-center">
          <Link to="/register" className="btn-primary px-10 py-4 text-base">
            Get Started Free <TbArrowRight />
          </Link>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default About;
