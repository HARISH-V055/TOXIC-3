from pydantic import BaseModel, Field
from typing import Optional, List, Tuple


class PredictRequest(BaseModel):
    """Input schema for toxicity prediction."""
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="SMILES representation of the molecule",
        examples=["CC(=O)Oc1ccccc1C(=O)O"],
    )


class BondAttention(BaseModel):
    atom_a: int = Field(..., description="Index of the first atom")
    atom_b: int = Field(..., description="Index of the second atom")
    weight: float = Field(..., ge=0.0, le=1.0, description="Attention weight of this bond")


class PredictResponse(BaseModel):
    """Output schema for toxicity prediction."""
    prediction: str = Field(..., description="'toxic' or 'non-toxic'")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of toxicity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    important_atoms: List[int] = Field(default_factory=list, description="Indices of high-attention atoms")
    important_bonds: List[BondAttention] = Field(default_factory=list, description="High-attention bonds")
    execution_time: float = Field(..., description="Inference time in milliseconds")


class ExplainRequest(BaseModel):
    """Input schema for GNN attention explanation."""
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="SMILES string to explain",
    )
    prediction_id: Optional[str] = Field(
        None,
        description="Optional reference to a stored prediction",
    )


class AtomAttention(BaseModel):
    atom_index: int
    weight: float = Field(..., ge=0.0, le=1.0)


class BondAttentionDetail(BaseModel):
    bond_index: int
    weight: float = Field(..., ge=0.0, le=1.0)
    atoms: Tuple[int, int]


class ExplainResponse(BaseModel):
    """Output schema for GNN attention explanation."""
    atom_attentions: List[AtomAttention] = Field(default_factory=list)
    bond_attentions: List[BondAttentionDetail] = Field(default_factory=list)
    saliency_map: List[List[float]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    message: str
