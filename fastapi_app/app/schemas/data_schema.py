from pydantic import BaseModel, Field

class UploadRequest(BaseModel):
    place_id: int
    upload_secret_key: str = Field(..., description="업로드 요청을 위한 인증 키")