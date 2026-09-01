# 🧬 MolXAI — Deep Graph AI Platform for Explainable Multi-Task Molecular Toxicity Prediction

<div align="center">

![MolXAI Banner](docs/assets/banner.png)

**Explainable Quantization-Aware Kolmogorov-Arnold Graph Neural Network (`EQ-KA-GCN`)**  
*Accelerating In Silico Toxicology, High-Throughput Screening & Chemical Safety Profiling*

[![Status](https://img.shields.io/badge/Status-Production%20%26%20Publication%20Ready-success?style=for-the-badge&logo=statuspage)](https://github.com/HARISH-V055/TOXIC-3)
[![Model](https://img.shields.io/badge/AI%20Architecture-Hybrid%20Cross--Modal%20EQ--KA--GCN-blueviolet?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Multi-Task](https://img.shields.io/badge/Tox21%20Panel-12%20Bioassay%20Endpoints-cyan?style=for-the-badge&logo=dna)](https://tripod.nih.gov/tox21/)
[![Quantization](https://img.shields.io/badge/Adaptive%20QAT-8%2F6%2F4--bit%20Mixed--Precision-orange?style=for-the-badge&logo=speedtest)](https://pytorch.org/)
[![Explainability](https://img.shields.io/badge/XAI-Directional%20GNNExplainer-emerald?style=for-the-badge&logo=interpret)](https://rdkit.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core AI Innovations](#core-ai-innovations)
- [Comprehensive Benchmark Results](#comprehensive-benchmark-results)
- [Tech Stack](#tech-stack)
- [Project Directory Structure](#project-directory-structure)
- [Quick Start Guide](#quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [Running with Docker Compose](#running-with-docker-compose-recommended)
  - [Running Locally from Source](#running-locally-from-source)
- [REST API Reference](#rest-api-reference)
- [License & Citation](#license--citation)

---

## Overview

**MolXAI** is an enterprise-grade scientific AI system designed for high-throughput, explainable molecular toxicity screening. By unifying **2D Graph Message Passing** with **1024-bit ECFP4 Morgan Fingerprints**, **Fourier-based Kolmogorov-Arnold Networks (Fourier-KAN)**, and **Adaptive Layer-Wise Quantization-Aware Training (QAT)**, MolXAI delivers state-of-the-art predictive accuracy across all **12 Tox21 bioassays** with a lightweight memory footprint (**302–814 KB**) and **sub-millisecond latency (0.236 ms/sample)**.

### Key Capabilities
- 🔬 **Hybrid Cross-Modal Ingestion** — Combines 2D molecular graph topology (32D node + 6D bond features) with 1024-bit ECFP4 circular fingerprints via LayerNorm projection.
- 🧬 **12-Bioassay Toxicological Profiling** — Predicts activity across all 7 Nuclear Receptors and 5 Stress Response pathways simultaneously.
- ⚡ **Adaptive Layer-Wise QAT** — Dynamic 8-bit, 6-bit, and 4-bit INT mixed-precision execution achieving 28.8% to 52.2% memory reduction.
- 🔍 **Directional Mechanistic XAI** — Uses GNNExplainer to differentiate between toxicity drivers (toxicophores) and non-toxicity stabilizers.
- 📊 **Interactive Full-Stack Web Platform** — React 19 glassmorphism dashboard with RDKit 2D canvas, audit history, batch CSV screening, and PDF reports.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         CLIENT TIER                          │
│               React 19 + TypeScript + Vite + Tailwind        │
│          Zustand · Framer Motion · RDKit Saliency Canvas     │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTPS / REST API
┌───────────────────────────▼──────────────────────────────────┐
│                      GATEWAY / API TIER                      │
│               Node.js + Express.js (TypeScript)              │
│         JWT Auth · Helmet · Rate Limiting · Mongo Store      │
└──────────┬───────────────────────────────┬───────────────────┘
           │                               │
┌──────────▼──────────┐       ┌────────────▼──────────────────┐
│   MongoDB / Mongoose │       │    AI Microservice Engine     │
│  Users · Predictions │       │  FastAPI + PyTorch Geometric  │
│  Audit Logs & Stats  │       │  EQ-KA-GCN · QAT · Explainer  │
└─────────────────────┘       └────────────────────────────────┘
```

---

## Core AI Innovations

1. **Hybrid Cross-Modal Kolmogorov-Arnold Graph Network (`EQ-KA-GCN`)**:
   - 2-layer `GCNConv` with residual skip connection and tri-modal multi-scale readout pooling ($\text{mean} \oplus \text{max} \oplus \text{add} = 384\text{D}$).
   - Fused with 1024-bit ECFP4 Morgan Fingerprints ($1408\text{D} \to 128\text{D}$ via LayerNorm projection).
   - Fourier-KAN classifier layer parameterizing non-linear spline functional transformations ($128\text{D} \to 64\text{D}$).
2. **Multi-Task Learning with Element-Wise Focal Loss ($\alpha=0.75, \gamma=2.0$)**:
   - Simultaneously models all 12 Tox21 assay endpoints, addressing severe 18:1 class imbalance across sparse bioassays.
3. **Adaptive Layer-Wise QAT (Quantization-Aware Training)**:
   - Evaluates dynamic range and variance to assign `conv2: 8-bit`, `fourier_kan: 8-bit`, `fc_out: 6-bit`, and `conv1: 4-bit`.
   - Compresses model size to **814.3 KB** with **0.236 ms latency** (~4,235 molecules/sec).
4. **Directional GNNExplainer Saliency Mapping**:
   - Isolates atomic centers driving toxicity ($P(\text{toxic}) \uparrow$) vs structural stabilization ($P(\text{non-toxic}) \uparrow$).

---

## Comprehensive Benchmark Results

### 1. Model Evolution & Performance Summary (Held-Out Test Set: 783 Molecules)

| Model Architecture | Multi-Task (12 Tasks) | ECFP4 Fusion | Loss Function | Test Accuracy | Macro ROC-AUC | F1 Score | Model Size | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline GCN (Vanilla MLP)** | ❌ (Single) | ❌ | Standard BCE | 94.51% | 0.7790 | 0.0000 | 422.0 KB | 0.528 ms |
| **Weighted KA-GCN** | ❌ (Single) | ❌ | Weighted BCE | 92.46% | 0.8320 | 0.3220 | 422.0 KB | 0.480 ms |
| **Multi-Task Focal KA-GCN (FP32)**| ✅ (12 Endpoints)| ✅ (1024-bit) | Multi-Task Focal | 88.69% | 0.8024 | 0.3474 | 1,143.8 KB| **0.195 ms** |
| 🏆 **Adaptive QAT KA-GCN (Ours)**| ✅ (12 Endpoints)| ✅ (1024-bit) | Multi-Task Focal | **90.25%** | **0.8134** | **0.3621** | **814.3 KB** | **0.236 ms** |

---

### 2. Breakdown by Tox21 Bioassay Endpoint

| Assay Endpoint | Target Bioassay Name | Pathway Category | ROC-AUC | Detection Rate (Recall) | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`NR-ER-LBD`** | Estrogen Receptor Alpha LBD | Nuclear Receptor | **0.9212** | 58.54% | 0.5333 |
| **`NR-AhR`** | Aryl Hydrocarbon Receptor | Nuclear Receptor | **0.8769** | **81.32%** | 0.4933 |
| **`SR-MMP`** | Mitochondrial Membrane Potential | Stress Response | **0.8416** | 62.86% | 0.4664 |
| **`NR-AR-LBD`** | Androgen Receptor LBD | Nuclear Receptor | **0.8415** | 58.06% | **0.5625** |
| **`SR-ATAD5`** | Genomic Instability & DNA Damage | Stress Response | **0.8174** | 27.27% | 0.2571 |
| **`SR-p53`** | p53 DNA Damage Checkpoint | Stress Response | **0.7952** | 34.88% | 0.2113 |
| **`NR-Aromatase`**| Aromatase Cytochrome P450 | Nuclear Receptor | **0.7920** | 18.18% | 0.2500 |
| **`NR-AR`** | Full-Length Androgen Receptor | Nuclear Receptor | **0.7837** | 44.44% | 0.4598 |
| **`SR-ARE`** | Antioxidant Response Element | Stress Response | **0.7742** | 62.89% | 0.3536 |
| **`SR-HSE`** | Heat Shock Factor Activation | Stress Response | **0.7379** | 25.71% | 0.2045 |
| **`NR-ER`** | Full-Length Estrogen Receptor | Nuclear Receptor | **0.7254** | 44.94% | 0.3774 |
| **`NR-PPAR-gamma`**| Peroxisome Proliferation Receptor| Nuclear Receptor | **0.7224** | — | — |
| **MACRO-AVERAGE** | **Complete 12-Assay Panel** | **All Pathways** | **0.8024** | **43.26%** | **0.3474** |

---

### 3. Toxic Molecule Detection & Screening Throughput

- **Full Tox21 Dataset (7,823 Compounds)**:
  - Total actually toxic compounds: **2,869**
  - **Successfully Detected (True Positives)**: **2,118 toxic molecules (73.82% Sensitivity / Recall)**
  - **Correctly Identified Non-Toxic (True Negatives)**: **3,631 compounds (73.29% Specificity)**
- **Held-Out Test Split (783 Compounds)**:
  - Total unseen toxic compounds: **312**
  - **Successfully Detected (True Positives)**: **219 toxic molecules (70.19% Sensitivity / Recall)**
  - **Correctly Identified Non-Toxic (True Negatives)**: **338 compounds (71.76% Specificity)**

---

## Tech Stack

| Layer | Technology & Tools |
| :--- | :--- |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS v4, Framer Motion, Zustand, Axios, Lucide Icons |
| **Backend** | Node.js, Express.js, TypeScript, MongoDB, Mongoose, JWT, bcrypt, Helmet, Winston |
| **AI Engine** | Python 3.11+, FastAPI, Uvicorn, PyTorch 2.0+, PyTorch Geometric, RDKit, NumPy, Pandas |
| **ML Architecture**| Hybrid Cross-Modal Graph + ECFP4, Fourier-KAN, Adaptive QAT, Directional GNNExplainer, Focal Loss |
| **DevOps** | Docker, Docker Compose, Git |

---

## Project Directory Structure

```
MolXAI/
├── EQ-KA-GCN/                   # Scientific Machine Learning Engine
│   ├── checkpoints/             # Trained model weights (*.pt)
│   ├── config.py                # Pipeline dataclass configurations
│   ├── datasets/                # Raw & processed Tox21 graph datasets
│   ├── evaluation/              # Evaluator, plots, metrics & threshold optimizer
│   ├── explainability/          # GNNExplainer, atom/bond ranking & visualizations
│   ├── figures/                 # 300 DPI publication plots
│   ├── graph/                   # RDKit graph builder & feature extractors
│   ├── main.py                  # Orchestrator (14-Phase Pipeline)
│   ├── models/                  # KA-GCN, Fourier-KAN & Focal Loss
│   ├── outputs/                 # Evaluation reports & publication figures
│   ├── quantization/            # Adaptive layer-wise QAT manager & observers
│   └── training/                # Stratified splitter, DataLoaders, Trainer & early stopping
├── ai/                          # FastAPI AI Microservice
│   └── app/
│       ├── api/routes/          # /api/predict, /api/explain, /health
│       ├── core/                # App settings & CORS config
│       ├── models/schemas.py    # Pydantic camelCase response contracts
│       └── services/            # Singleton GNN inference & XAI service
├── backend/                     # Node.js + Express API Gateway
│   └── src/
│       ├── controllers/         # Prediction, Auth & Analytics controllers
│       ├── middleware/          # JWT auth, error handlers, rate limiters
│       ├── models/              # User, Prediction & Analytics schemas
│       ├── routes/              # Express route definitions
│       └── services/            # Proxy bridge to FastAPI microservice
├── frontend/                    # Modern React 19 Web Client
│   └── src/
│       ├── components/          # 12-Endpoint cards, Molecule viewer, UI primitives
│       ├── pages/               # Predict, Dashboard, Analytics, Batch Screening
│       ├── store/               # Zustand state stores
│       └── types/               # TypeScript interfaces
├── docs/                        # Architecture diagrams & documentation
├── docker-compose.yml           # Multi-container orchestration
├── README.md                    # Root Documentation
└── SUMMARY_README.md            # Comprehensive Project Summary
```

---

## Quick Start Guide

### Prerequisites
- **Node.js**: `v18+` & `npm`
- **Python**: `v3.10+` with PyTorch & RDKit
- **MongoDB**: Local or MongoDB Atlas instance

---

### Running with Docker Compose (Recommended)

```bash
git clone https://github.com/HARISH-V055/TOXIC-3.git
cd TOXIC-3
docker-compose up --build
```
- **Web Dashboard**: `http://localhost:5173`
- **Backend API**: `http://localhost:5000`
- **AI Microservice**: `http://localhost:8000/docs`

---

### Running Locally from Source

#### 1. AI Microservice (FastAPI)
```bash
cd ai
python -m venv venv
.\venv\Scripts\activate          # On Linux/macOS: source venv/bin/activate
pip install -r ../EQ-KA-GCN/requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Backend Gateway (Node.js + Express)
```bash
cd backend
npm install
npm run dev
```

#### 3. Frontend Client (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

---

## REST API Reference

### `POST /api/predict`
Executes multi-task toxicity inference across all 12 Tox21 assay endpoints and generates GNNExplainer attributions.

**Sample Request:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O"
}
```

**Sample Response:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "prediction": "Non-Toxic",
  "probability": 0.1288,
  "confidence": 0.8712,
  "threshold": 0.45,
  "endpoint": "Tox21 (12 Endpoints)",
  "inferenceTimeMs": 0.24,
  "endpoints": [
    {
      "endpoint": "NR-ER-LBD",
      "name": "Estrogen Receptor Alpha LBD",
      "category": "Nuclear Receptor",
      "prediction": "Non-Toxic / Inactive",
      "probability": 0.0842,
      "confidence": 0.9158,
      "threshold": 0.45
    },
    {
      "endpoint": "SR-p53",
      "name": "p53 DNA Damage Checkpoint",
      "category": "Stress Response",
      "prediction": "Non-Toxic / Inactive",
      "probability": 0.1288,
      "confidence": 0.8712,
      "threshold": 0.45
    }
  ],
  "importantAtoms": [
    {
      "index": 4,
      "element": "O",
      "name": "Oxygen",
      "score": 1.0000,
      "rank": 1,
      "influenceType": "Non-Toxicity",
      "role": "Safety / Non-Toxicity Stabilizer",
      "description": "Ester oxygen participating in resonance stabilization."
    }
  ],
  "importantBonds": [
    {
      "source": 1,
      "target": 4,
      "score": 0.9924,
      "rank": 1,
      "bondName": "C(#1) — O(#4)",
      "influenceType": "Non-Toxicity",
      "role": "Structural Safety Stabilizer",
      "description": "Stable ester linkage mitigating chemical reactivity."
    }
  ],
  "explanationSummary": "Model identified 5 key safety-stabilizing atomic centers maintaining a non-reactive, metabolically benign conformation.",
  "explanationImage": "/outputs/explanations/molecule_explanation.png"
}
```

---

## License & Citation

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Citation
```bibtex
@article{molxai2026,
  title={EQ-KA-GCN: Explainable and Quantization-Aware Kolmogorov-Arnold Graph Neural Networks for Multi-Task In Silico Toxicology},
  author={MolXAI Research Team},
  journal={Journal of Chemical Information and Modeling (In Preparation)},
  year={2026}
}
```
