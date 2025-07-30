from pydantic import BaseModel

class TextEmbeddingRequest(BaseModel):
    text: str
    user_id: str