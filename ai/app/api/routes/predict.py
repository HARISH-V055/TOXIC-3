from fastapi import APIRouter, HTTPException, status
from app.models.schemas import PredictRequest, PredictResponse
from app.services.gnn_service import gnn_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Molecular Toxicity",
    description="Accepts a SMILES string and returns a toxicity prediction with confidence scores and atom/bond importance weights from the EQ-KA-GCN model.",
)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        logger.info(f"Prediction request received for SMILES: {request.smiles[:30]}")
        result = gnn_service.predict(request)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid SMILES string: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference failed. Please try again.",
        )
