import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ProtectedRoute } from './ProtectedRoute';
import { DashboardLayout } from '@layouts/DashboardLayout';
import { AuthLayout } from '@layouts/AuthLayout';
import { Spinner } from '@components/ui/Spinner';

// Lazy load pages for optimal bundle splitting
const Landing = lazy(() => import('@pages/Landing'));
const Login = lazy(() => import('@pages/Login'));
const Register = lazy(() => import('@pages/Register'));
const Dashboard = lazy(() => import('@pages/Dashboard'));
const Predict = lazy(() => import('@pages/Predict'));
const History = lazy(() => import('@pages/History'));
const Profile = lazy(() => import('@pages/Profile'));
const Settings = lazy(() => import('@pages/Settings'));
const About = lazy(() => import('@pages/About'));
const Documentation = lazy(() => import('@pages/Documentation'));
const Research = lazy(() => import('@pages/Research'));
const Architecture = lazy(() => import('@pages/Architecture'));
const Experiments = lazy(() => import('@pages/Experiments'));
const Benchmark = lazy(() => import('@pages/Benchmark'));
const NotFound = lazy(() => import('@pages/NotFound'));

const PageLoader = () => (
  <div className="flex items-center justify-center min-h-[50vh]">
    <Spinner size="lg" text="Loading..." />
  </div>
);

export const AppRoutes: React.FC = () => {
  return (
    <Suspense fallback={<PageLoader />}>
      <Routes>
        {/* Public routes */}
        <Route path="/" element={<Landing />} />
        <Route path="/about" element={<About />} />
        <Route path="/documentation" element={<Documentation />} />

        {/* Auth routes */}
        <Route element={<AuthLayout />}>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
        </Route>

        {/* Protected dashboard routes */}
        <Route element={<ProtectedRoute />}>
          <Route element={<DashboardLayout />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/history" element={<History />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/research" element={<Research />} />
            <Route path="/architecture" element={<Architecture />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/benchmark" element={<Benchmark />} />
          </Route>
        </Route>

        {/* Fallback */}
        <Route path="/404" element={<NotFound />} />
        <Route path="*" element={<Navigate to="/404" replace />} />
      </Routes>
    </Suspense>
  );
};
