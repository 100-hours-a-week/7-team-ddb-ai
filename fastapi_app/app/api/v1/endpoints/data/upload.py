from fastapi import APIRouter, Depends, HTTPException, status, Body

from app.core.config import settings
from app.api.deps import get_data_uploader
from app.schemas.data_schema import UploadRequest
from app.data_pipeline.pipeline import UploaderPipeline

router = APIRouter()

@router.post(
    "/upload",
    status_code=status.HTTP_200_OK,
    summary="장소 데이터 추가",
    description="장소 id를 기준으로 새로운 장소를 추가합니다"
)
async def upload_data(
    req: UploadRequest = Body(..., description="오늘의 추천 장소"),
    uploader: UploaderPipeline = Depends(get_data_uploader)
) -> dict:
    if req.upload_secret_key != settings.UPLOAD_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid secret key"
        )
    
    try:
        uploader.upload_data(place_id=req.place_id)
        return {"message": "Upload completed"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )