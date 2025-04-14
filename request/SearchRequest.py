from pydantic import BaseModel
from typing import List

class SearchRequest(BaseModel):
    account_id: int
    doc_id: int
    interactive: int


class MultipleSearchRequest(BaseModel):
    searchs: List[SearchRequest]