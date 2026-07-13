from fastapi import APIRouter, HTTPException, status
from app.models.schemas import ExplainRequest, ExplainResponse
from app.services.gnn_service import gnn_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/explain",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Explain GNN Prediction",
    description="Returns attention-based explainability data: per-atom and per-bond weights, and optional saliency maps from the GNN model.",
)
async def explain(request: ExplainRequest) -> ExplainResponse:
    try:
        logger.info(f"Explain request received for SMILES: {request.smiles[:30]}")
        result = gnn_service.explain(request)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid SMILES string: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explanation generation failed. Please try again.",
        )
