from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DocumentRequest(BaseModel):
    document_id: str | int
    title: str
    category: str
    content: Optional[str]
    popularity: float

class UserRequest(BaseModel):
    account_id: str | int

class UserInteractionRequest(BaseModel):
    account_id: str | int
    document_id: str | int
    timestamp: datetime
    view_time: float
    rating: int | None

class MultipleNewDataRequest(BaseModel):
    updateDocuments: List[DocumentRequest]
    newDocuments: List[DocumentRequest]
    userInteractions: List[UserInteractionRequest]
    userRequests: List[UserRequest]