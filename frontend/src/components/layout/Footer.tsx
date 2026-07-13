import React from 'react';
import { Link } from 'react-router-dom';
import { TbAtom } from 'react-icons/tb';
import { FiGithub, FiTwitter } from 'react-icons/fi';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-white/5 bg-surface-850/50 py-8 mt-auto">
      <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <TbAtom className="text-primary-400 text-xl" />
          <span className="text-sm font-semibold gradient-text">MolXAI</span>
          <span className="text-white/20 text-sm">·</span>
          <span className="text-white/40 text-xs">Molecular Toxicity AI</span>
        </div>
        <div className="flex items-center gap-4 text-xs text-white/40">
          <Link to="/about" className="hover:text-white/70 transition-colors">About</Link>
          <Link to="/documentation" className="hover:text-white/70 transition-colors">Docs</Link>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-white/70 transition-colors"
            aria-label="GitHub"
          >
            <FiGithub />
          </a>
          <a
            href="https://twitter.com"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-white/70 transition-colors"
            aria-label="Twitter"
          >
            <FiTwitter />
          </a>
        </div>
        <p className="text-xs text-white/25">
          © {new Date().getFullYear()} MolXAI. All rights reserved.
        </p>
      </div>
    </footer>
  );
};
