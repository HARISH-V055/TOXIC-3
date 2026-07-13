import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { TbAtom, TbBook, TbCpu, TbTerminal } from 'react-icons/tb';
import { MdCode } from 'react-icons/md';
import { Footer } from '@components/layout/Footer';

const endpoints = [
  { method: 'POST', path: '/api/auth/register', desc: 'Register a new user account', auth: false },
  { method: 'POST', path: '/api/auth/login', desc: 'Login and receive access token', auth: false },
  { method: 'POST', path: '/api/auth/logout', desc: 'Logout and clear refresh token', auth: true },
  { method: 'POST', path: '/api/auth/refresh', desc: 'Refresh access token via HttpOnly cookie', auth: 'cookie' },
  { method: 'GET', path: '/api/user/profile', desc: 'Get current user profile', auth: true },
  { method: 'PUT', path: '/api/user/profile', desc: 'Update profile name/email', auth: true },
  { method: 'PUT', path: '/api/user/password', desc: 'Change user password', auth: true },
  { method: 'DELETE', path: '/api/user', desc: 'Delete account and all data', auth: true },
  { method: 'POST', path: '/api/predictions', desc: 'Submit SMILES for toxicity prediction', auth: true },
  { method: 'GET', path: '/api/predictions', desc: 'Get prediction history (paginated)', auth: true },
  { method: 'GET', path: '/api/predictions/:id', desc: 'Get a specific prediction by ID', auth: true },
  { method: 'GET', path: '/api/predictions/stats', desc: 'Get aggregate prediction statistics', auth: true },
  { method: 'DELETE', path: '/api/predictions/:id', desc: 'Delete a prediction by ID', auth: true },
  { method: 'GET', path: '/api/ai/status', desc: 'Get AI model and service status', auth: false },
  { method: 'POST', path: '/api/ai/predict', desc: 'Direct AI inference proxy (SMILES → prediction)', auth: true },
  { method: 'POST', path: '/api/ai/explain', desc: 'GNN attention explanation proxy', auth: true },
];

const methodColors: Record<string, string> = {
  GET: 'text-green-400 bg-green-500/10 border-green-500/20',
  POST: 'text-primary-400 bg-primary-500/10 border-primary-500/20',
  PUT: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20',
  DELETE: 'text-red-400 bg-red-500/10 border-red-500/20',
};

const Documentation: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'api' | 'setup' | 'model'>('api');

  return (
    <div className="min-h-screen bg-surface-900 bg-grid-pattern text-white">
      <header className="border-b border-white/5 px-6 py-4 flex items-center gap-3 backdrop-blur-md bg-surface-900/80 sticky top-0 z-10">
        <Link to="/" className="flex items-center gap-2">
          <TbAtom className="text-primary-400 text-xl" />
          <span className="font-bold gradient-text">MolXAI</span>
        </Link>
        <span className="text-white/20">/</span>
        <span className="text-sm text-white/50">Documentation</span>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-12 space-y-12">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-black text-white mb-2 flex items-center gap-3">
              <TbBook className="text-primary-400" /> Platform Documentation
            </h1>
            <p className="text-white/40 text-sm">Deployment steps, dataset details, and API references.</p>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 p-1.5 rounded-xl bg-white/3 border border-white/5">
            <button
              onClick={() => setActiveTab('api')}
              className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'api' ? 'bg-primary-500 text-white' : 'text-white/40 hover:text-white/70'
              }`}
            >
              REST API
            </button>
            <button
              onClick={() => setActiveTab('setup')}
              className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'setup' ? 'bg-primary-500 text-white' : 'text-white/40 hover:text-white/70'
              }`}
            >
              Installation
            </button>
            <button
              onClick={() => setActiveTab('model')}
              className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'model' ? 'bg-primary-500 text-white' : 'text-white/40 hover:text-white/70'
              }`}
            >
              Model & Training
            </button>
          </div>
        </div>

        {/* ─── TAB 1: API Reference ──────────────────────────────── */}
        {activeTab === 'api' && (
          <div className="space-y-8">
            <div className="glass-card p-5">
              <p className="text-xs text-white/40 mb-2 uppercase tracking-widest">Base URL</p>
              <code className="font-mono text-primary-300 text-sm">http://localhost:5000</code>
              <p className="text-[11px] text-white/30 mt-2">
                All endpoints are prefixed with <code className="text-white/60">/api</code>
              </p>
            </div>

            <div className="glass-card p-5">
              <h2 className="text-base font-semibold text-white flex items-center gap-2 mb-3">
                <MdCode /> Authentication
              </h2>
              <p className="text-xs text-white/50 mb-3">
                Protected endpoints require a Bearer token in the Authorization header:
              </p>
              <pre className="bg-black/40 rounded-xl p-4 text-[11px] font-mono text-primary-300 overflow-x-auto">
{`Authorization: Bearer <access_token>

# Refresh tokens are managed via secure HttpOnly cookies`}
              </pre>
            </div>

            <div>
              <h2 className="text-lg font-bold text-white mb-4">API Endpoints</h2>
              <div className="glass-card overflow-hidden p-0">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/5">
                      <th className="text-left text-xs font-semibold text-white/40 px-5 py-3">Method</th>
                      <th className="text-left text-xs font-semibold text-white/40 px-5 py-3">Endpoint</th>
                      <th className="text-left text-xs font-semibold text-white/40 px-5 py-3 hidden md:table-cell">Description</th>
                      <th className="text-center text-xs font-semibold text-white/40 px-5 py-3">Auth</th>
                    </tr>
                  </thead>
                  <tbody>
                    {endpoints.map(({ method, path, desc, auth }) => (
                      <tr key={path + method} className="border-b border-white/5 hover:bg-white/2 transition-colors">
                        <td className="px-5 py-3">
                          <span className={`inline-flex px-2 py-0.5 rounded-md text-xs font-bold border ${methodColors[method]}`}>
                            {method}
                          </span>
                        </td>
                        <td className="px-5 py-3">
                          <code className="text-xs font-mono text-white/70">{path}</code>
                        </td>
                        <td className="px-5 py-3 text-xs text-white/45 hidden md:table-cell">{desc}</td>
                        <td className="px-5 py-3 text-center text-xs">
                          {auth === true ? (
                            <span className="text-primary-400">🔒 JWT</span>
                          ) : auth === 'cookie' ? (
                            <span className="text-yellow-400">🍪 Cookie</span>
                          ) : (
                            <span className="text-white/25">Public</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ─── TAB 2: Setup & Deployment ────────────────────────── */}
        {activeTab === 'setup' && (
          <div className="space-y-6 text-xs text-white/50 leading-relaxed">
            {/* Frontend */}
            <div className="glass-card p-6">
              <h3 className="text-base font-semibold text-white mb-3 flex items-center gap-2">
                <TbTerminal className="text-primary-400" /> Frontend Configuration
              </h3>
              <p className="mb-3">React 19 Vite application setup instructions:</p>
              <pre className="bg-black/40 p-4 rounded-xl font-mono text-[11px] text-white/70 overflow-x-auto">
{`cd frontend
npm install
npm run dev

# Starts on http://localhost:5173`}
              </pre>
            </div>

            {/* Backend */}
            <div className="glass-card p-6">
              <h3 className="text-base font-semibold text-white mb-3 flex items-center gap-2">
                <TbTerminal className="text-accent-400" /> Backend Setup
              </h3>
              <p className="mb-3">Node.js API gateway configuration. Requires a local MongoDB running on default port 27017:</p>
              <pre className="bg-black/40 p-4 rounded-xl font-mono text-[11px] text-white/70 overflow-x-auto">
{`cd backend
npm install
npm run dev

# Starts on http://localhost:5000`}
              </pre>
            </div>

            {/* FastAPI */}
            <div className="glass-card p-6">
              <h3 className="text-base font-semibold text-white mb-3 flex items-center gap-2">
                <TbTerminal className="text-violet-400" /> AI Microservice (FastAPI)
              </h3>
              <p className="mb-3">Python FastAPI environment with the learnable Spline convolutions:</p>
              <pre className="bg-black/40 p-4 rounded-xl font-mono text-[11px] text-white/70 overflow-x-auto">
{`cd ai
py -3.12 -m venv venv
.\\venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Starts on http://localhost:8000`}
              </pre>
            </div>
          </div>
        )}

        {/* ─── TAB 3: Model & Training ───────────────────────────── */}
        {activeTab === 'model' && (
          <div className="space-y-6 text-xs text-white/50 leading-relaxed">
            <div className="glass-card p-6">
              <h3 className="text-base font-semibold text-white mb-3 flex items-center gap-2">
                <TbCpu className="text-primary-400" /> EQ-KA-GCN Convolutions
              </h3>
              <p className="mb-3">
                By replacing conventional matrix products $W \cdot x$ inside layers with 1D splines, 
                the model parameters act as learnable spline interpolation paths on edges.
              </p>
              <p>
                This allows the neural networks to capture subtle geometric transformations (e.g. angle variances, aromatic structures) 
                much more robustly compared to baseline models.
              </p>
            </div>

            <div className="glass-card p-6">
              <h3 className="text-base font-semibold text-white mb-3">Model Quantization Calibration</h3>
              <p className="mb-3">
                Quantization scales model parameters to INT8 limitations, bypassing float32 memory bounds on mobile edge devices. 
                Using QAT, clipping points are evaluated during gradient calculations.
              </p>
              <p>
                This results in a final serialized model size of just **18MB**, down from over **120MB** for uncompressed counterparts.
              </p>
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
};

export default Documentation;
