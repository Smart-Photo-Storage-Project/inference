# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from app.models import TextEmbeddingRequest
from app.embedding_service import get_image_embedding, get_text_embedding
import numpy as np

app = FastAPI()

@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        embedding = get_image_embedding(contents)
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/text")
async def embed_text(payload: TextEmbeddingRequest):
    try:
        embedding = get_text_embedding(payload.text)
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
