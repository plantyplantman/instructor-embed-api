from pydantic import BaseModel
from typing import Optional, List


class EmbassPayload(BaseModel):
    texts: List[str]
    instruction: Optional[str] = "Represent the query for retrieval"
