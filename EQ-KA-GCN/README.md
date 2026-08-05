# 🧬 EQ-KA-GCN

**Explainable Quantization-Aware Graph Convolutional Network with Fourier-KAN for Molecular Toxicity Prediction**

This repository contains the scientific implementation of the EQ-KA-GCN framework designed for molecular toxicity prediction. It replaces traditional Multi-Layer Perceptrons (MLPs) in Graph Convolutional Networks with Kolmogorov-Arnold spline convolutions (Fourier-KAN) and integrates Quantization-Aware Training (QAT) for edge device optimizations.

---

## 📁 Folder Structure

```
EQ-KA-GCN/
├── README.md             # Project documentation and setup guide
├── requirements.txt      # Python dependencies list
├── .gitignore            # Git exclusion rules
├── config.py             # Global dataclass configuration module
├── main.py               # Main entry point (project initialization)
├── datasets/             # Data directory
│   ├── raw/              # Unprocessed raw SMILES files
│   └── processed/        # PyTorch Geometric graph datasets
├── graph/                # Graph representation & construction modules
├── models/               # Graph neural network model architectures
├── training/             # Model training pipelines and loops
├── evaluation/           # Evaluation metrics and validation scripts
├── quantization/         # Quantization-Aware Training (QAT) modules
├── explainability/       # Explainable AI (XAI) attention visualization
├── utils/                # Project utility modules
│   ├── logger.py         # Persistent file and console logging pipeline
│   ├── seed.py           # Seed setup for scientific reproducibility
│   ├── device.py         # Automatically hardware-detection utils
│   └── helpers.py        # Custom helpers and metric formatting
├── checkpoints/          # Saved model weights/checkpoints
├── outputs/              # Evaluation metrics and prediction output logs
├── figures/              # Figures for IEEE publication drafts
└── logs/                 # Persistent execution and training log files
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11 or later
- PyTorch and PyTorch Geometric compatible system

### 1. Create a Virtual Environment
Navigate to the project root directory `EQ-KA-GCN/` and run the command matching your operating system:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Ensure your virtual environment is active, then run:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

To execute the project initialization, run the following command in your terminal:

```bash
python main.py
```

This will run the initialization pipeline which:
- Configures random seeds across all libraries (`random`, `numpy`, `torch`, `CUDA`)
- Automatically detects the computing hardware (e.g. CPU or CUDA-capable GPU)
- Resolves all directory path locations
- Generates a reusable logger and logs the initialization metadata to `logs/training.log` and the terminal
