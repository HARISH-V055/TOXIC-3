import React from 'react';
import { Link } from 'react-router-dom';
import { MdMenu, MdScience } from 'react-icons/md';
import { useAuthStore } from '@store/useAuthStore';

interface NavbarProps {
  onMenuClick: () => void;
  pageTitle?: string;
}

export const Navbar: React.FC<NavbarProps> = ({ onMenuClick, pageTitle }) => {
  const user = useAuthStore((s) => s.user);

  return (
    <header
      className="
        fixed top-0 right-0 left-0 lg:left-[260px] z-20 h-16
        bg-surface-850/80 backdrop-blur-md border-b border-white/5
        flex items-center px-4 lg:px-6 gap-4
      "
    >
      {/* Mobile menu toggle */}
      <button
        onClick={onMenuClick}
        className="lg:hidden p-2 rounded-lg text-white/60 hover:text-white hover:bg-white/5 transition-colors"
        aria-label="Open menu"
        id="navbar-menu-btn"
      >
        <MdMenu className="text-xl" />
      </button>

      {/* Page title */}
      {pageTitle && (
        <div className="flex-1 hidden sm:block">
          <h2 className="text-sm font-semibold text-white/70">{pageTitle}</h2>
        </div>
      )}

      <div className="flex-1 sm:flex-none" />

      {/* Right actions */}
      <div className="flex items-center gap-2">
        {/* Quick predict CTA */}
        <Link
          to="/predict"
          id="navbar-predict-link"
          className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold
            bg-gradient-to-r from-primary-500/20 to-accent-500/20
            border border-primary-500/30 text-primary-300
            hover:border-primary-400/50 hover:text-primary-200 transition-all duration-200"
        >
          <MdScience className="text-sm" />
          New Prediction
        </Link>

        {/* Avatar */}
        <Link
          to="/profile"
          id="navbar-profile-link"
          className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shrink-0 hover:opacity-80 transition-opacity"
          aria-label="Profile"
        >
          <span className="text-xs font-bold text-white">
            {user?.name?.charAt(0)?.toUpperCase() ?? 'U'}
          </span>
        </Link>
      </div>
    </header>
  );
};
