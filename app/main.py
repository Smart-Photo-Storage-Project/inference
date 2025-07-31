# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from app.models import TextEmbeddingRequest
from app.embedding_service import get_text_embedding, search_similar_images, embed_and_store_images_batch
from app.qdrant import client, collection_name, init_qdrant
from app.api_key import verify_api_key
from sentence_transformers import SentenceTransformer
from typing import List
from PIL import Image
from io import BytesIO

app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_qdrant()

@app.on_event("startup")
async def load_models():
    app.state.img_model = SentenceTransformer("clip-ViT-B-32")
    app.state.text_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

@app.post("/embed/images")
async def embed_images(
    request: Request,
    files: List[UploadFile] = File(...),
    names: List[str] = Form(...),
    user_id: str = Form(...),
    upload_at: int = Form(...),
    paths: List[str] = Form(...)
):
    if not (len(files) == len(names) == len(paths)):
        raise HTTPException(status_code=400, detail="Mismatch in lengths of files, names, and paths.")

    try:
        file_bytes_batch = [await f.read() for f in files]

        points = embed_and_store_images_batch(
            batch_file_bytes=file_bytes_batch,
            names=names,
            user_id=user_id,
            upload_at=upload_at,
            paths=paths,
            request=request
        )

        return {
            "status": "stored in qdrant",
            "stored_count": len(points),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/embed/text", dependencies=[Depends(verify_api_key)])
async def embed_text(request: Request, payload: TextEmbeddingRequest):
    try:
        embedding = get_text_embedding(payload.text, request)
        results = search_similar_images(embedding.tolist(), user_id=payload.user_id, top_k=12)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
