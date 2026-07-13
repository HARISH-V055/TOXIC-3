import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MdDashboard,
  MdScience,
  MdHistory,
  MdPerson,
  MdSettings,
  MdInfo,
  MdMenuBook,
  MdLogout,
  MdClose,
  MdLayers,
  MdTimeline,
  MdLeaderboard,
  MdBook,
} from 'react-icons/md';
import { TbAtom } from 'react-icons/tb';
import { useAuth } from '@hooks/useAuth';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const navItems = [
  { to: '/dashboard', icon: MdDashboard, label: 'Dashboard' },
  { to: '/predict', icon: MdScience, label: 'Predict' },
  { to: '/history', icon: MdHistory, label: 'History' },
];

const researchItems = [
  { to: '/research', icon: MdBook, label: 'Research' },
  { to: '/architecture', icon: MdLayers, label: 'Architecture' },
  { to: '/experiments', icon: MdTimeline, label: 'Experiments' },
  { to: '/benchmark', icon: MdLeaderboard, label: 'Benchmark' },
];

const userItems = [
  { to: '/profile', icon: MdPerson, label: 'Profile' },
  { to: '/settings', icon: MdSettings, label: 'Settings' },
];

const infoItems = [
  { to: '/about', icon: MdInfo, label: 'About' },
  { to: '/documentation', icon: MdMenuBook, label: 'Documentation' },
];

export const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <>
      {/* Mobile Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-30 lg:hidden"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ x: isOpen ? 0 : '-100%' }}
        transition={{ type: 'spring', damping: 30, stiffness: 300 }}
        className="sidebar lg:translate-x-0"
      >
        {/* Logo */}
        <div className="flex items-center justify-between px-4 py-5 border-b border-white/5">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shadow-glow-sm">
              <TbAtom className="text-white text-xl" />
            </div>
            <div>
              <h1 className="text-base font-bold gradient-text">MolXAI</h1>
              <p className="text-[10px] text-white/40 -mt-0.5">Toxicity Prediction AI</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="lg:hidden p-1.5 rounded-lg text-white/40 hover:text-white hover:bg-white/5 transition-colors"
            aria-label="Close sidebar"
          >
            <MdClose />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          <p className="px-3 text-[10px] font-semibold uppercase tracking-wider text-white/25 mb-2">
            Main
          </p>
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              onClick={onClose}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <Icon className="text-lg shrink-0" />
              <span>{label}</span>
            </NavLink>
          ))}

          <div className="my-3 divider" />

          <p className="px-3 text-[10px] font-semibold uppercase tracking-wider text-white/25 mb-2">
            Research & Dev
          </p>
          {researchItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              onClick={onClose}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <Icon className="text-lg shrink-0" />
              <span>{label}</span>
            </NavLink>
          ))}

          <div className="my-3 divider" />

          <p className="px-3 text-[10px] font-semibold uppercase tracking-wider text-white/25 mb-2">
            Account
          </p>
          {userItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              onClick={onClose}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <Icon className="text-lg shrink-0" />
              <span>{label}</span>
            </NavLink>
          ))}

          <div className="my-3 divider" />

          <p className="px-3 text-[10px] font-semibold uppercase tracking-wider text-white/25 mb-2">
            Info
          </p>
          {infoItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              onClick={onClose}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <Icon className="text-lg shrink-0" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* User Profile Footer */}
        <div className="px-3 py-4 border-t border-white/5">
          <div className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/3 mb-2">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shrink-0">
              <span className="text-xs font-bold text-white">
                {user?.name?.charAt(0)?.toUpperCase() ?? 'U'}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-white truncate">{user?.name ?? 'User'}</p>
              <p className="text-[10px] text-white/40 truncate">{user?.email ?? ''}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="sidebar-link w-full text-red-400/70 hover:text-red-400 hover:bg-red-500/5"
          >
            <MdLogout className="text-lg" />
            <span>Logout</span>
          </button>
        </div>
      </motion.aside>
    </>
  );
};
