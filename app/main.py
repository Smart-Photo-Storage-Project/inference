# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from app.models import TextEmbeddingRequest
from app.embedding_service import get_text_embedding, search_similar_images, embed_and_store_image
from app.qdrant import client, collection_name, init_qdrant
from sentence_transformers import SentenceTransformer

app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_qdrant()

@app.on_event("startup")
async def load_models():
    app.state.img_model = SentenceTransformer("clip-ViT-B-32")
    app.state.text_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

@app.post("/embed/image")
async def embed_image(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
    user_id: str = Form(...),
    upload_at: int = Form(...),
    path: str = Form(...)):
    try:
        contents = await file.read()
        point_id, metadata = embed_and_store_image(contents, name, user_id, upload_at, path, request)

        return {
            "status": "stored in qdrant",
            "point_id": point_id,
            "metadata": metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/text")
async def embed_text(request: Request, payload: TextEmbeddingRequest):
    try:
        embedding = get_text_embedding(payload.text, request)
        results = search_similar_images(embedding.tolist(), top_k=5)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
