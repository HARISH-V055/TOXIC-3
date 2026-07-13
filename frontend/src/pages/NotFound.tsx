import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { TbAtom } from 'react-icons/tb';
import { MdHome, MdArrowBack } from 'react-icons/md';

const NotFound: React.FC = () => {
  return (
    <div className="min-h-screen bg-surface-900 bg-grid-pattern flex items-center justify-center p-6 text-white">
      <div className="absolute top-0 left-0 right-0 h-[400px] bg-glow-cyan pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center max-w-lg relative z-10"
      >
        <motion.div
          animate={{ rotate: [0, 10, -10, 0] }}
          transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          className="mb-8"
        >
          <TbAtom className="text-[120px] text-primary-500/20 mx-auto" />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-8xl font-black gradient-text">404</span>
          </div>
        </motion.div>

        <h1 className="text-2xl font-bold text-white mb-3">Molecule Not Found</h1>
        <p className="text-white/40 mb-8 text-sm leading-relaxed">
          This page doesn't exist in our molecular database. Perhaps try a different bond pathway?
        </p>

        <div className="flex items-center justify-center gap-3 flex-wrap">
          <Link to="/" className="btn-primary px-6 py-3" id="404-home-btn">
            <MdHome /> Go Home
          </Link>
          <button
            onClick={() => window.history.back()}
            className="btn-secondary px-6 py-3"
            id="404-back-btn"
          >
            <MdArrowBack /> Go Back
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default NotFound;
