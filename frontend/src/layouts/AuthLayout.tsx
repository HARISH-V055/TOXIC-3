import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { TbAtom } from 'react-icons/tb';

export const AuthLayout: React.FC = () => {
  return (
    <div className="min-h-screen bg-surface-900 bg-grid-pattern flex items-center justify-center p-4 relative overflow-hidden">
      {/* Background glows */}
      <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-primary-500/5 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-accent-500/5 rounded-full blur-3xl pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="w-full max-w-md relative z-10"
      >
        {/* Logo */}
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-3 mb-2">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shadow-glow-cyan">
              <TbAtom className="text-white text-2xl" />
            </div>
            <div className="text-left">
              <h1 className="text-2xl font-bold gradient-text">MolXAI</h1>
              <p className="text-xs text-white/40">Molecular Toxicity AI</p>
            </div>
          </Link>
        </div>

        <Outlet />

        <p className="text-center text-xs text-white/25 mt-8">
          © {new Date().getFullYear()} MolXAI. All rights reserved.
        </p>
      </motion.div>
    </div>
  );
};
