from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScriptGenerateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    theme: str = Field(..., min_length=1)
    style: str = Field(default="甜宠")
    duration_min: int = Field(default=3, ge=1, le=5)
    keywords: List[str] = Field(default_factory=list)


class ScriptUpdateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    script_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    title: Optional[str] = None


class ShotGenerateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    script_id: Optional[str] = None
    content: Optional[str] = None


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    selected_doc_id: Optional[str] = None


class UploadDocResponse(BaseModel):
    file_name: str
    chunks: int


class ExportRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    script_id: str = Field(..., min_length=1)
    format: str = Field(default="txt")


class ToolRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    script_id: Optional[str] = None
    content: Optional[str] = None
    instruction: str = Field(default="")


class MetaRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    script_id: Optional[str] = None
    content: Optional[str] = None


class AutoPlanRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1)
    brief: Optional[str] = None
    script_id: Optional[str] = None
    selected_doc_id: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=20)


class ApiResponse(BaseModel):
    code: int = 200
    msg: str = "ok"
    data: Optional[Dict[str, Any]] = None
