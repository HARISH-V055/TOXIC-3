# 🧬 MolXAI — AI-Powered Molecular Toxicity Prediction System

<div align="center">
  <img src="docs/assets/banner.png" alt="MolXAI Banner" />
  <br />
  <p><strong>Graph Neural Network-powered molecular toxicity analysis with explainability</strong></p>
  <img src="https://img.shields.io/badge/Status-Architecture%20Ready-blue" />
  <img src="https://img.shields.io/badge/AI%20Model-EQ--KA--GCN-cyan" />
  <img src="https://img.shields.io/badge/Backend-Node.js%20%2B%20Express-green" />
  <img src="https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite-orange" />
  <img src="https://img.shields.io/badge/Database-MongoDB-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Running Locally](#running-locally)
  - [Running with Docker](#running-with-docker)
- [API Reference](#api-reference)
- [AI Model Integration](#ai-model-integration)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

MolXAI is a production-ready, full-stack AI SaaS application for molecular toxicity prediction. It implements **EQ-KA-GCN (Explainable Quantization-Aware Kolmogorov-Arnold Graph Convolutional Network)** to predict whether a molecule (represented in SMILES format) is toxic or non-toxic, along with detailed explainability through attention weights on atoms and bonds.

### Key Features

- 🔬 **SMILES Input** — Standard chemical notation input for molecular structures
- 🤖 **EQ-KA-GCN Convolutions** — Kolmogorov-Arnold spline convolutions on edges replacing MLPs
- ⚡ **Quantization-Aware (QAT)** — INT8 quantization calibration for hardware-efficient edge execution
- 🎯 **Confidence Scoring** — Probability and confidence estimates per prediction
- 🔍 **Explainability** — Atom and bond importance highlighting from GNN attention
- 📊 **Prediction History** — Full audit trail per user with search and filtering
- 🔐 **JWT Authentication** — Secure user authentication with refresh token rotation
- 📱 **Responsive UI** — Dark-mode, glassmorphism-styled React dashboard

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                         │
│               React 19 + Vite + Tailwind CSS                 │
│            Zustand · Framer Motion · React Router            │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTPS / REST API
┌───────────────────────────▼──────────────────────────────────┐
│                         API LAYER                            │
│               Node.js + Express.js (TypeScript)              │
│         JWT Auth · Helmet · Rate Limiting · Morgan           │
└──────────┬───────────────────────────────┬───────────────────┘
           │                               │
┌──────────▼──────────┐       ┌────────────▼──────────────────┐
│   MongoDB / Mongoose │       │    AI Microservice Layer      │
│  Users · Predictions │       │  FastAPI + PyTorch Geometric  │
│  ModelInformation    │       │  EQ-KA-GCN · RDKit · NumPy    │
└─────────────────────┘       └────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Vite, TypeScript, Tailwind CSS v4, Framer Motion, Zustand, React Router DOM v6, Axios, React Hook Form, React Icons |
| **Backend** | Node.js, Express.js, TypeScript, MongoDB, Mongoose, JWT, bcrypt, Helmet, Morgan, CORS, Multer, express-validator |
| **AI Service** | Python, FastAPI, Uvicorn, PyTorch, PyTorch Geometric, RDKit, NumPy, Pandas |
| **DevOps** | Docker, Docker Compose, GitHub Actions (CI/CD ready) |

---

## Project Structure

```
MolXAI/
├── frontend/                  # React + Vite SPA
│   ├── src/
│   │   ├── assets/            # Static images, icons
│   │   ├── components/        # Reusable UI components
│   │   │   ├── ui/            # Base UI primitives
│   │   │   ├── layout/        # Sidebar, Navbar, Footer
│   │   │   ├── molecule/      # Molecule visualization
│   │   │   ├── prediction/    # Prediction result components
│   │   │   └── dashboard/     # Dashboard-specific components
│   │   ├── hooks/             # Custom React hooks
│   │   ├── layouts/           # Page layout wrappers
│   │   ├── pages/             # Route-level page components
│   │   ├── routes/            # Route configuration & guards
│   │   ├── services/          # API service layer (Axios)
│   │   ├── store/             # Zustand state management
│   │   ├── styles/            # Global CSS, Tailwind config
│   │   ├── types/             # TypeScript type definitions
│   │   └── utils/             # Helper utilities
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── backend/                   # Express API Service
│   ├── src/
│   │   ├── config/            # Database, env config
│   │   ├── controllers/       # Route controllers
│   │   ├── middleware/        # Auth, errors, rate limit
│   │   ├── models/            # Mongoose schemas
│   │   ├── routes/            # Express routers
│   │   ├── services/          # Business logic
│   │   ├── types/             # TypeScript types
│   │   ├── utils/             # Logger, response helpers
│   │   ├── validators/        # express-validator schemas
│   │   ├── app.ts             # Express app setup
│   │   └── server.ts          # HTTP server entry point
│   ├── package.json
│   └── tsconfig.json
│
├── ai/                        # Python AI Microservice (placeholder)
│   ├── app/
│   │   ├── api/routes/        # Predict & explain endpoints
│   │   ├── core/              # Configuration settings
│   │   ├── models/            # Pydantic schemas
│   │   └── services/          # GNN inference service (placeholder)
│   ├── requirements.txt
│   └── Dockerfile
│
├── docs/                      # Documentation
│   └── api.md                 # API specification
│
├── docker/                    # Docker build files
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   └── ai.Dockerfile
│
├── docker-compose.yml         # Multi-service orchestration
└── README.md
```

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- **Node.js** >= 20.x
- **npm** >= 10.x
- **MongoDB** >= 7.x (local or Atlas URI)
- **Python** >= 3.11 (for AI service)
- **Docker** & **Docker Compose** (optional, for containerized setup)

---

### Environment Variables

**Backend** — copy `backend/.env.example` to `backend/.env` and fill in your values:

```bash
cp backend/.env.example backend/.env
```

**Frontend** — copy `frontend/.env.example` to `frontend/.env`:

```bash
cp frontend/.env.example frontend/.env
```

---

### Running Locally

#### 1. Backend

```bash
cd backend
npm install
npm run dev        # starts on http://localhost:5000
```

#### 2. Frontend

```bash
cd frontend
npm install
npm run dev        # starts on http://localhost:5173
```

#### 3. AI Service (Placeholder)

```bash
cd ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

### Running with Docker

The entire stack can be launched with a single command:

```bash
docker compose up --build
```

This will start:
- **MongoDB** on port `27017`
- **Backend API** on port `5000`
- **Frontend** on port `5173`
- **AI Service** on port `8000`

---

## API Reference

See [docs/api.md](docs/api.md) for the full API specification.

### Quick Reference

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/auth/register` | Register new user | Public |
| POST | `/api/auth/login` | Login user | Public |
| POST | `/api/auth/logout` | Logout user | Private |
| POST | `/api/auth/refresh` | Refresh access token | Cookie |
| GET | `/api/user/profile` | Get current user profile | Private |
| PUT | `/api/user/profile` | Update user profile | Private |
| DELETE | `/api/user` | Delete account | Private |
| POST | `/api/predictions` | Submit SMILES for prediction | Private |
| GET | `/api/predictions` | Get prediction history | Private |
| DELETE | `/api/predictions/:id` | Delete a prediction | Private |
| POST | `/api/ai/predict` | AI inference proxy | Private |
| POST | `/api/ai/explain` | GNN attention explain proxy | Private |

---

## AI Model Integration

The AI microservice (`/ai`) is fully scaffolded and ready for integration. To integrate the EQ-KA-GCN model:

1. Place your trained model weights in `ai/app/models/weights/`
2. Update `ai/app/services/gnn_service.py` with model loading and inference logic
3. The backend's `aiService.ts` already has the HTTP proxy calls configured
4. Set `AI_SERVICE_URL` in your backend `.env`

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'feat: add my feature'`
4. Push to branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.
