"""
Configuration Module for EQ-KA-GCN

Centralizes all configurable parameters for the project, including file paths,
hyperparameters, training configurations, and quantization settings.
Uses Python dataclasses with type hints and supports easy adjustments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union


@dataclass
class PathConfig:
    """Configuration for directory paths."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.raw_dir = self.base_dir / "datasets" / "raw"
        self.processed_dir = self.base_dir / "datasets" / "processed"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.outputs_dir = self.base_dir / "outputs"
        self.figures_dir = self.base_dir / "figures"
        self.logs_dir = self.base_dir / "logs"


@dataclass
class ModelConfig:
    """Hyperparameters for the baseline GCN model."""
    name: str = "BaselineGCN"
    save_filename: str = "eq_ka_gcn_best.pt"
    input_dim: int = 5  # 5 node features (atomic number, degree, formal charge, aromaticity, hydrogens)
    hidden_dim: int = 128
    output_dim: int = 1  # Binary prediction (toxic/non-toxic)
    dropout: float = 0.3
    num_gcn_layers: int = 2



@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    seed: int = 42
    batch_size: int = 64
    lr: float = 0.001
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 15
    early_stopping: int = 15
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1



@dataclass
class QuantizationConfig:
    """Configuration for Quantization-Aware Training (QAT)."""
    enabled: bool = True
    qat_enabled: bool = True
    default_bits: int = 8
    bits: int = 8
    adaptive: bool = True
    supported_bits: List[int] = field(default_factory=lambda: [4, 6, 8])
    calibration_batches: int = 10
    calibration_batch_limit: int = 10
    qat_save_filename: str = "eq_ka_gcn_qat_best.pt"
    qat_history_filename: str = "qat_history.csv"
    qat_report_filename: str = "quantization_report.json"


@dataclass
class DataConfig:
    """Configuration for dataset parameters."""
    raw_filename: str = "tox21.csv"
    processed_filename: str = "tox21_processed.csv"
    graphs_filename: str = "graphs.pt"
    info_filename: str = "dataset_info.json"
    train_graphs_filename: str = "train_graphs.pt"
    val_graphs_filename: str = "validation_graphs.pt"
    test_graphs_filename: str = "test_graphs.pt"
    smiles_column: str = "SMILES"
    target_column: str = "SR-p53"





@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    threshold: float = 0.5
    save_plots: bool = True
    save_reports: bool = True


@dataclass
class FourierKANConfig:
    """Configuration for Fourier-KAN (Kolmogorov-Arnold Network) classifier."""
    enabled: bool = True
    fourier_order: int = 5
    hidden_dim: int = 64
    dropout: float = 0.2
    activation: str = "silu"
    save_filename: str = "ka_gcn_best.pt"
    history_filename: str = "ka_gcn_history.csv"
    use_weighted_loss: bool = True
    weighted_save_filename: str = "ka_gcn_weighted_best.pt"
    weighted_history_filename: str = "ka_gcn_weighted_history.csv"


@dataclass
class ThresholdConfig:
    """Configuration for decision threshold optimization."""
    enabled: bool = True
    search_start: float = 0.05
    search_end: float = 0.95
    step: float = 0.05
    selection_metric: str = "f1"


@dataclass
class ExplainabilityConfig:
    """Configuration for GNNExplainer module."""
    enabled: bool = True
    top_k_atoms: int = 5
    top_k_bonds: int = 5
    save_visualization: bool = True
    save_report: bool = True
    explanations_dir: str = "outputs/explanations"


@dataclass
class Config:
    """Master configuration class."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fourier_kan: FourierKANConfig = field(default_factory=FourierKANConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    device: str = "auto"  # options: 'auto', 'cuda', 'cpu'


def get_config() -> Config:
    """Factory function to retrieve the default configuration."""
    return Config()


