# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from app.models import TextEmbeddingRequest
from app.embedding_service import get_image_embedding, get_text_embedding
import numpy as np

app = FastAPI()

@app.post("/embed/image")
async def embed_image(
    file: UploadFile = File(...),
    name: str = Form(...),
    user_id: str = Form(...),
    upload_at: int = Form(...),):
    try:
        contents = await file.read()
        embedding = get_image_embedding(contents)

        return {
            "embedding": embedding.tolist(),
            "name": name,
            "user_id": user_id,
            "upload_at": upload_at,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/text")
async def embed_text(payload: TextEmbeddingRequest):
    try:
        embedding = get_text_embedding(payload.text)
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
