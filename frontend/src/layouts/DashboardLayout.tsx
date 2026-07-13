import React, { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Sidebar } from '@components/layout/Sidebar';
import { Navbar } from '@components/layout/Navbar';

const pageTitles: Record<string, string> = {
  '/dashboard': 'Dashboard',
  '/predict': 'New Prediction',
  '/history': 'Prediction History',
  '/profile': 'My Profile',
  '/settings': 'Settings',
  '/about': 'About MolXAI',
  '/documentation': 'Documentation',
  '/research': 'Research Overview',
  '/architecture': 'System Architecture',
  '/experiments': 'Experimental Results',
  '/benchmark': 'Benchmark Comparison',
};

export const DashboardLayout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();
  const pageTitle = pageTitles[location.pathname];

  return (
    <div className="min-h-screen bg-surface-900 bg-grid-pattern">
      <div className="absolute inset-0 bg-glow-cyan pointer-events-none" />

      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <Navbar onMenuClick={() => setSidebarOpen(true)} pageTitle={pageTitle} />

      <main
        className="lg:ml-[260px] pt-16 min-h-screen"
        id="main-content"
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className="p-4 md:p-6 lg:p-8 max-w-7xl mx-auto"
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
};
