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
    name: Optional[str] = Field(None, description="Full element name (e.g. Oxygen)")
    score: float = Field(..., description="GNNExplainer importance score")
    rank: Optional[int] = Field(None, description="Importance rank (1 = highest)")
    influence_type: Optional[str] = Field(None, description="'Toxicity' or 'Non-Toxicity'")
    role: Optional[str] = Field(None, description="Role: 'Toxicity Driver (Toxicophore)' or 'Safety / Non-Toxicity Stabilizer'")
    description: Optional[str] = Field(None, description="Chemical mechanistic explanation")


class ImportantBond(BaseCamelModel):
    source: int = Field(..., description="Source atom index")
    target: int = Field(..., description="Target atom index")
    score: float = Field(..., description="GNNExplainer importance score")
    rank: Optional[int] = Field(None, description="Importance rank")
    bond_name: Optional[str] = Field(None, description="Bond label e.g. 'C(#1) — O(#2)'")
    influence_type: Optional[str] = Field(None, description="'Toxicity' or 'Non-Toxicity'")
    role: Optional[str] = Field(None, description="Role: 'Toxicity-Propagating Bond' or 'Structural Safety Stabilizer'")
    description: Optional[str] = Field(None, description="Chemical mechanistic description")


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


class EndpointPrediction(BaseCamelModel):
    endpoint: str = Field(..., description="Endpoint identifier (e.g. NR-AR, SR-p53)")
    name: str = Field(..., description="Full biological target name")
    category: str = Field(..., description="'Nuclear Receptor' or 'Stress Response'")
    prediction: str = Field(..., description="'Toxic / Active' or 'Non-Toxic / Inactive'")
    probability: float = Field(..., ge=0.0, le=1.0, description="Assay activity probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    threshold: float = Field(default=0.5, description="Decision threshold")


class PredictResponse(BaseCamelModel):
    """Output schema for toxicity prediction."""
    smiles: str = Field(..., description="SMILES string")
    prediction: str = Field(..., description="'Toxic' or 'Non-Toxic'")
    probability: float = Field(..., ge=0.0, le=1.0, description="Toxicity probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    threshold: float = Field(default=0.5, description="Classification decision threshold")
    endpoint: str = Field(default="Tox21 (12 Endpoints)", description="Target bioassay endpoint")
    inference_time_ms: float = Field(..., description="Model inference time in milliseconds")
    endpoints: List[EndpointPrediction] = Field(default_factory=list, description="Predictions across all 12 Tox21 assay endpoints")
    important_atoms: List[ImportantAtom] = Field(default_factory=list)
    important_bonds: List[ImportantBond] = Field(default_factory=list)
    explanation_summary: Optional[str] = Field(default=None, description="Mechanistic explanation of atoms influencing toxicity or non-toxicity")
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
    target_endpoint: Optional[str] = Field(
        default="SR-p53",
        description="Specific endpoint to explain (e.g. SR-p53, NR-AR, etc.)",
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


