# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from app.models import TextEmbeddingRequest
from app.embedding_service import get_image_embedding, get_text_embedding
from app.qdrant import client, collection_name, init_qdrant
from qdrant_client.models import PointStruct
import numpy as np
import uuid

app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_qdrant()

@app.post("/embed/image")
async def embed_image(
    file: UploadFile = File(...),
    name: str = Form(...),
    user_id: str = Form(...),
    upload_at: int = Form(...),):
    try:
        contents = await file.read()
        embedding = get_image_embedding(contents)

        payload = {
            "name": name,
            "user_id": user_id,
            "upload_at": upload_at
        }

        point_id = str(uuid.uuid4())

        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            ]
        )

        return {
            "status": "stored in qdrant",
            "point_id": point_id,
            "metadata": payload
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
