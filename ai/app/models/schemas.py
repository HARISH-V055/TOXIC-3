from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel
from typing import Optional, List, Tuple


class BaseCamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )


class PredictRequest(BaseModel):
    """Input schema for toxicity prediction."""
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="SMILES representation of the molecule",
        examples=["CC(=O)Oc1ccccc1C(=O)O"],
    )


class ImportantAtom(BaseCamelModel):
    index: int = Field(..., description="Atom index")
    element: str = Field(..., description="Element symbol")
    score: float = Field(..., description="GNNExplainer importance score")


class ImportantBond(BaseCamelModel):
    source: int = Field(..., description="Source atom index")
    target: int = Field(..., description="Target atom index")
    score: float = Field(..., description="GNNExplainer importance score")


class GraphAtom(BaseCamelModel):
    index: int
    element: str
    x: float
    y: float


class GraphBond(BaseCamelModel):
    source: int
    target: int


class MolecularGraph(BaseCamelModel):
    atoms: List[GraphAtom] = Field(default_factory=list)
    bonds: List[GraphBond] = Field(default_factory=list)


class PredictResponse(BaseCamelModel):
    """Output schema for toxicity prediction."""
    smiles: str = Field(..., description="SMILES string")
    prediction: str = Field(..., description="'Toxic' or 'Non-Toxic'")
    probability: float = Field(..., ge=0.0, le=1.0, description="Toxicity probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    threshold: float = Field(default=0.75, description="Classification decision threshold")
    endpoint: str = Field(default="Tox21 SR-p53", description="Target bioassay endpoint")
    inference_time_ms: float = Field(..., description="Model inference time in milliseconds")
    important_atoms: List[ImportantAtom] = Field(default_factory=list)
    important_bonds: List[ImportantBond] = Field(default_factory=list)
    molecular_graph: MolecularGraph = Field(default_factory=MolecularGraph)
    explanation_image: str = Field(default="/outputs/explanations/molecule_explanation.png")


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


class AtomAttention(BaseCamelModel):
    atom_index: int
    weight: float = Field(..., ge=0.0, le=1.0)


class BondAttentionDetail(BaseCamelModel):
    bond_index: int
    weight: float = Field(..., ge=0.0, le=1.0)
    atoms: Tuple[int, int]


class ExplainResponse(BaseCamelModel):
    """Output schema for GNN attention explanation."""
    atom_attentions: List[AtomAttention] = Field(default_factory=list)
    bond_attentions: List[BondAttentionDetail] = Field(default_factory=list)
    saliency_map: List[List[float]] = Field(default_factory=list)


class HealthResponse(BaseCamelModel):
    status: str
    version: str
    model_loaded: bool
    message: str


